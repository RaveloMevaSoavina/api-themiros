"""Extraction du texte (spec §4) : PDF natif, scanné ou mixte, DOCX, XLSX.

Le texte est conservé page par page, intact (RG-4.6). Un PDF scanné passe
par l'OCR Tesseract page par page, en parallèle. Le texte suit l'ordre de
lecture : colonnes d'une page PDF lues l'une après l'autre, tableaux à leur
place, valeurs d'un classeur rattachées à leur en-tête de colonne.
"""

import logging
import zipfile
from collections import defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import date, datetime
from io import BytesIO
from typing import Literal
from xml.etree import ElementTree

import cv2
import numpy as np
import pymupdf
import pytesseract
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from app.core.errors import ApiError
from app.modules.ingestion.errors import (
    corrupted_file,
    ocr_unavailable,
    password_protected,
    unsupported_format,
)

logger = logging.getLogger(__name__)

DocumentFormat = Literal["pdf", "docx", "xlsx"]
PdfType = Literal["natif", "scan", "mixte"]

# Spec §4.2 : une page de moins de 100 caractères qui contient une image est
# une page scannée.
SCANNED_PAGE_MAX_CHARS = 100
WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
# Une page en deux colonnes : chaque colonne porte au moins 20 % du texte,
# les blocs à cheval sur la gouttière (titre, encadré) au plus 30 %.
COLUMN_MIN_SHARE = 0.2
SPANNING_MAX_SHARE = 0.3
# En-tête d'un tableau de classeur : cherché dans les premières lignes.
XLSX_HEADER_SCAN_ROWS = 10


@dataclass(frozen=True, slots=True)
class OcrOptions:
    languages: str = "fra+eng"
    dpi: int = 300
    page_timeout_seconds: int = 120
    parallelism: int = 4


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    page_number: int
    text: str
    section_title: str | None = None
    is_ocr: bool = False
    ocr_failed: bool = False
    char_start: int = 0
    # Titres repérés par leur style (DOCX), dans l'ordre du document.
    headings: tuple[str, ...] = ()
    # Confiance moyenne de l'OCR (0-1), None sans OCR ou si l'OCR a échoué.
    ocr_confidence: float | None = None

    def is_exploitable(self, min_chars: int) -> bool:
        return len(self.text.strip()) >= min_chars


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    format: DocumentFormat
    pages: list[ExtractedPage]
    pdf_type: PdfType | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    def exploitable_pages(self, min_chars: int) -> int:
        return sum(page.is_exploitable(min_chars) for page in self.pages)


def detect_format(content: bytes, filename: str) -> DocumentFormat:
    """Le contenu fait foi, pas l'extension déclarée."""
    if content.startswith(b"%PDF"):
        return "pdf"
    if content.startswith(b"PK"):
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile as error:
            raise corrupted_file(filename) from error
        if "word/document.xml" in names:
            return "docx"
        if "xl/workbook.xml" in names:
            return "xlsx"
    raise unsupported_format(filename)


def extract_document(
    content: bytes, filename: str, ocr: OcrOptions | None = None
) -> ExtractedDocument:
    document_format = detect_format(content, filename)
    try:
        if document_format == "pdf":
            extracted = _extract_pdf(content, ocr or OcrOptions())
        elif document_format == "docx":
            extracted = _extract_docx(content)
        else:
            extracted = _extract_xlsx(content)
    except ApiError:
        raise
    except Exception as error:
        raise corrupted_file(filename) from error
    return _with_offsets(extracted)


def detect_pdf_type(texts: Sequence[str], has_images: Sequence[bool]) -> PdfType:
    """Algorithme de la spec §4.2, sur le texte natif de chaque page."""
    if sum(len(text) for text in texts) < SCANNED_PAGE_MAX_CHARS * max(1, len(texts)):
        return "scan"
    if any(
        len(text) < SCANNED_PAGE_MAX_CHARS and images
        for text, images in zip(texts, has_images, strict=True)
    ):
        return "mixte"
    return "natif"


def _extract_pdf(content: bytes, ocr: OcrOptions) -> ExtractedDocument:
    document = pymupdf.open(stream=content, filetype="pdf")
    try:
        if document.needs_pass:
            raise password_protected()
        # Une seule lecture du texte natif : il sert au type du PDF et au
        # contenu des pages.
        natives: list[str] = []
        has_images: list[bool] = []
        pages: list[ExtractedPage] = []
        for index, page in enumerate(document):
            text = page.get_text("text", sort=True).strip()
            natives.append(text)
            has_images.append(bool(page.get_images()))
            # Un tableau sans texte natif n'a que des cellules vides.
            if text:
                text = _reading_text(page, text)
            pages.append(ExtractedPage(page_number=index + 1, text=text))
        pdf_type = detect_pdf_type(natives, has_images)
        to_ocr = [
            index
            for index, page in enumerate(pages)
            if len(page.text) < SCANNED_PAGE_MAX_CHARS
            and (has_images[index] or pdf_type == "scan")
        ]

        if to_ocr:
            # PyMuPDF n'est pas thread-safe : le rendu est séquentiel, l'OCR
            # (processus Tesseract) est parallélisé.
            images = (
                (index, _render_page(document[index], ocr.dpi)) for index in to_ocr
            )
            for index, result in _ocr_pages(images, ocr):
                pages[index] = replace(
                    pages[index],
                    text=result.text or pages[index].text,
                    is_ocr=True,
                    ocr_failed=result.failed,
                    ocr_confidence=result.confidence,
                )
        return ExtractedDocument("pdf", pages, pdf_type)
    finally:
        document.close()


Box = tuple[float, float, float, float]


def _reading_text(page: pymupdf.Page, text: str) -> str:
    """Texte d'une page dans son ordre de lecture.

    Les cellules d'un tableau figurent déjà dans le texte natif, sans leur
    structure : les blocs situés dans un tableau en sont retirés et le
    tableau structuré prend leur place. Une page en deux colonnes est lue
    colonne par colonne, et non ligne par ligne à travers la gouttière. Une
    page sans tableau ni colonnes garde son texte natif.
    """
    tables = _pdf_tables(page)
    items: list[tuple[Box, str]] = list(tables)
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, value, _, block_type = block[:7]
        value = value.strip()
        if block_type != 0 or not value:
            continue
        center = ((x0 + x1) / 2, (y0 + y1) / 2)
        if any(_contains(bbox, center) for bbox, _ in tables):
            continue
        items.append(((x0, y0, x1, y1), value))
    gutter = _gutter(items, page.rect)
    if not tables and gutter is None:
        return text
    return "\n\n".join(_reading_order(items, gutter))


def _gutter(items: Sequence[tuple[Box, str]], rect: pymupdf.Rect) -> float | None:
    """Abscisse de la gouttière d'une page en deux colonnes, sinon None.

    Chaque colonne doit porter du texte courant (un bloc d'au moins trois
    lignes) : une page de libellés et de valeurs alignés n'est pas une page
    en colonnes, et la lire colonne par colonne séparerait chaque libellé de
    sa valeur.
    """
    total = sum(len(value) for _, value in items)
    if total == 0 or len(items) < 2:
        return None
    best: tuple[int, float] | None = None
    for step in range(int(rect.width * 0.3), int(rect.width * 0.7) + 1, 2):
        x = rect.x0 + step
        left = [(box, value) for box, value in items if box[2] <= x]
        right = [(box, value) for box, value in items if box[0] >= x]
        left_chars = sum(len(value) for _, value in left)
        right_chars = sum(len(value) for _, value in right)
        crossing = total - left_chars - right_chars
        if (
            left_chars < total * COLUMN_MIN_SHARE
            or right_chars < total * COLUMN_MIN_SHARE
            or crossing > total * SPANNING_MAX_SHARE
            or not _has_running_text(left)
            or not _has_running_text(right)
        ):
            continue
        if best is None or crossing < best[0]:
            best = (crossing, x)
    return None if best is None else best[1]


def _has_running_text(items: Sequence[tuple[Box, str]]) -> bool:
    return any(value.count("\n") >= 2 for _, value in items)


def _reading_order(items: Sequence[tuple[Box, str]], gutter: float | None) -> list[str]:
    """De haut en bas ; sur deux colonnes, chaque bande comprise entre deux
    blocs pleine largeur est lue colonne gauche puis colonne droite."""
    ordered = sorted(items, key=lambda item: (item[0][1], item[0][0]))
    if gutter is None:
        return [value for _, value in ordered]
    texts: list[str] = []
    left: list[str] = []
    right: list[str] = []
    for box, value in ordered:
        if box[2] <= gutter:
            left.append(value)
        elif box[0] >= gutter:
            right.append(value)
        else:
            texts.extend(left + right)
            left, right = [], []
            texts.append(value)
    texts.extend(left + right)
    return texts


def _contains(bbox: Box, point: tuple[float, float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _pdf_tables(page: pymupdf.Page) -> list[tuple[Box, str]]:
    """Tableaux d'une page : emprise et lignes « cellule | cellule »."""
    # La stratégie par défaut (« lines ») ne trouve un tableau que dans les
    # tracés vectoriels : sans tracé, `find_tables` (~45 ms par page) est
    # évité.
    if not page.get_cdrawings():
        return []
    try:
        found = page.find_tables()
    except Exception:  # noqa: BLE001 — une page sans structure exploitable
        return []
    tables: list[tuple[Box, str]] = []
    for table in found.tables:
        rows = [
            " | ".join((cell or "").replace("\n", " ").strip() for cell in row)
            for row in table.extract()
        ]
        rows = [row for row in rows if row.strip(" |")]
        if rows:
            tables.append((tuple(table.bbox), "\n".join(rows)))
    return tables


def _render_page(page: pymupdf.Page, dpi: int) -> np.ndarray:
    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width
    )


@dataclass(frozen=True, slots=True)
class OcrResult:
    text: str
    failed: bool = False
    confidence: float | None = None


def _ocr_pages(
    images: Iterator[tuple[int, np.ndarray]], options: OcrOptions
) -> Iterator[tuple[int, OcrResult]]:
    with ThreadPoolExecutor(max_workers=options.parallelism) as pool:
        pending = []
        for index, image in images:
            pending.append((index, pool.submit(ocr_image, image, options)))
            # Borne la mémoire : on n'attend pas la fin du rendu pour lire.
            while len(pending) >= options.parallelism * 2:
                done_index, future = pending.pop(0)
                yield done_index, future.result()
        for index, future in pending:
            yield index, future.result()


def ocr_image(gray: np.ndarray, options: OcrOptions) -> OcrResult:
    """OCR d'une page (spec §4.4) : texte et confiance moyenne des mots."""
    image = preprocess_for_ocr(gray)
    try:
        data = pytesseract.image_to_data(
            image,
            lang=options.languages,
            config="--psm 3 --oem 1",
            timeout=options.page_timeout_seconds,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as error:
        raise ocr_unavailable() from error
    except (RuntimeError, pytesseract.TesseractError):
        # Spec §4.7 : OCR échoué → extraction partielle et alerte.
        logger.warning("OCR failed on one page", exc_info=True)
        return OcrResult("", failed=True)
    return ocr_result(data)


def ocr_result(data: dict[str, list]) -> OcrResult:
    """Texte des mots reconnus, une ligne par ligne Tesseract, une ligne vide
    entre paragraphes ; confiance moyenne pondérée par la longueur des mots."""
    paragraphs: dict[tuple[int, int], dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    weighted = 0.0
    letters = 0
    for index, word in enumerate(data.get("text", [])):
        word = (word or "").strip()
        if not word:
            continue
        paragraph = (int(data["block_num"][index]), int(data["par_num"][index]))
        paragraphs[paragraph][int(data["line_num"][index])].append(word)
        confidence = float(data["conf"][index])
        if confidence >= 0:
            weighted += confidence * len(word)
            letters += len(word)
    text = "\n\n".join(
        "\n".join(" ".join(words) for words in lines.values())
        for lines in paragraphs.values()
    )
    confidence = round(weighted / letters / 100, 3) if letters else None
    return OcrResult(text.strip(), confidence=confidence)


def preprocess_for_ocr(gray: np.ndarray) -> np.ndarray:
    """Niveaux de gris, binarisation d'Otsu, redressement, débruitage."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = deskew(binary)
    return cv2.medianBlur(binary, 3)


def deskew(binary: np.ndarray) -> np.ndarray:
    ink = np.column_stack(np.where(binary < 128))
    if len(ink) < 50:
        return binary
    angle = cv2.minAreaRect(ink.astype(np.float32))[-1]
    if angle > 45:
        angle -= 90
    # Au-delà, l'estimation est plus probablement fausse que la page penchée.
    if abs(angle) < 0.3 or abs(angle) > 15:
        return binary
    height, width = binary.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        binary,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _extract_docx(content: bytes) -> ExtractedDocument:
    """Paragraphes, titres, listes et tableaux dans l'ordre du document, puis
    notes de bas de page (spec §4.5). Un DOCX n'a pas de pages : une seule."""
    document = DocxDocument(BytesIO(content))
    blocks: list[str] = []
    headings: list[str] = []
    for child in document.element.body.iterchildren():
        if child.tag == f"{WORD_NAMESPACE}p":
            paragraph = Paragraph(child, document)
            value = paragraph.text.strip()
            if not value:
                continue
            style = (paragraph.style.name if paragraph.style else "") or ""
            if style.startswith(("Heading", "Titre", "Title")):
                # Un titre tient sur une ligne pour le découpage.
                value = " ".join(value.split())
                headings.append(value)
            elif style.startswith(("List", "Liste")):
                value = f"- {value}"
            blocks.append(value)
        elif child.tag == f"{WORD_NAMESPACE}tbl":
            table = Table(child, document)
            rows = [
                " | ".join(cell.text.strip() for cell in row.cells)
                for row in table.rows
            ]
            rows = [row for row in rows if row.strip(" |")]
            if rows:
                blocks.append("\n".join(rows))
    footnotes = _docx_footnotes(content)
    if footnotes:
        blocks.append("\n".join(footnotes))
    return ExtractedDocument(
        "docx",
        [
            ExtractedPage(
                page_number=1,
                text="\n\n".join(blocks),
                section_title=headings[0] if headings else None,
                headings=tuple(headings),
            )
        ],
    )


def _docx_footnotes(content: bytes) -> list[str]:
    with zipfile.ZipFile(BytesIO(content)) as archive:
        if "word/footnotes.xml" not in archive.namelist():
            return []
        root = ElementTree.fromstring(archive.read("word/footnotes.xml"))
    notes: list[str] = []
    for footnote in root.iter(f"{WORD_NAMESPACE}footnote"):
        if int(footnote.attrib.get(f"{WORD_NAMESPACE}id", "-1")) < 1:
            continue
        text = "".join(node.text or "" for node in footnote.iter(f"{WORD_NAMESPACE}t"))
        if text.strip():
            notes.append(text.strip())
    return notes


def _extract_xlsx(content: bytes) -> ExtractedDocument:
    """Une page par feuille, valeurs calculées des formules (spec §4.6).

    Sous la ligne d'en-tête d'un tableau, chaque valeur est rattachée à son
    en-tête (« B7 Budget : 120000 ») : « B7=120000 » seul ne dit pas ce que
    vaut le nombre. Les autres cellules restent repérées par leur coordonnée.
    """
    workbook = load_workbook(BytesIO(content), read_only=True, data_only=True)
    try:
        pages: list[ExtractedPage] = []
        for index, sheet in enumerate(workbook.worksheets, start=1):
            rows: list[tuple[int, dict[int, str]]] = []
            for row in sheet.iter_rows():
                values = {
                    cell.column: _cell_text(cell.value)
                    for cell in row
                    if getattr(cell, "value", None) is not None
                    and _cell_text(cell.value)
                }
                if values:
                    rows.append((row[0].row, values))
            pages.append(
                ExtractedPage(
                    page_number=index,
                    text="\n".join(_sheet_lines(rows)),
                    section_title=sheet.title,
                )
            )
        return ExtractedDocument("xlsx", pages)
    finally:
        workbook.close()


def _sheet_lines(rows: Sequence[tuple[int, dict[int, str]]]) -> list[str]:
    header_position = next(
        (
            position
            for position, (_, values) in enumerate(rows[:XLSX_HEADER_SCAN_ROWS])
            if _is_header(values)
        ),
        None,
    )
    headers = rows[header_position][1] if header_position is not None else {}
    lines: list[str] = []
    for position, (number, values) in enumerate(rows):
        if header_position is None or position <= header_position:
            # Titre de la feuille, puis la ligne d'en-tête elle-même.
            lines.append(
                " | ".join(
                    f"{get_column_letter(column)}{number}={value}"
                    for column, value in values.items()
                )
            )
            continue
        lines.append(
            " | ".join(
                f"{get_column_letter(column)}{number} {headers[column]} : {value}"
                if column in headers
                else f"{get_column_letter(column)}{number}={value}"
                for column, value in values.items()
            )
        )
    return lines


def _is_header(values: dict[int, str]) -> bool:
    """Au moins deux libellés, aucun nombre."""
    return len(values) >= 2 and not any(_is_number(value) for value in values.values())


def _is_number(value: str) -> bool:
    try:
        float(value.replace(" ", "").replace(",", "."))
    except ValueError:
        return False
    return True


def _cell_text(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, datetime) and value.time() == datetime.min.time():
        return value.date().isoformat()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return " ".join(str(value).split())


def _with_offsets(document: ExtractedDocument) -> ExtractedDocument:
    """Position de chaque page dans le texte complet (pages jointes par une
    ligne vide), pour les `char_start` / `char_end` des segments."""
    offset = 0
    pages: list[ExtractedPage] = []
    for page in document.pages:
        pages.append(replace(page, char_start=offset))
        offset += len(page.text) + 2
    return replace(document, pages=pages)
