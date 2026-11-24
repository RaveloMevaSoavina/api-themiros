"""PDF d'une facture manuelle (série TH-AAAA-NNNN, saisie au backoffice).

Les factures Stripe ont leur propre PDF chez Stripe ; celui-ci reprend la
même mise en page que l'export des rapports (`exports/renderers.render_pdf`) :
HTML rendu par pymupdf, sans dépendance de plus.
"""

import io
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from html import escape
from typing import Any

import pymupdf

CSS = """
* { font-family: sans-serif; }
body { font-size: 10pt; color: #1f2937; }
h1 { font-size: 18pt; margin: 0 0 4pt 0; }
.muted { color: #6b7280; font-size: 9pt; }
.parties { margin-top: 18pt; }
.party { margin-bottom: 10pt; }
.party b { display: block; margin-bottom: 2pt; }
table { width: 100%; border-collapse: collapse; margin-top: 18pt; }
th { text-align: left; border-bottom: 1px solid #9ca3af; padding: 4pt 0; }
td { padding: 6pt 0; border-bottom: 1px solid #e5e7eb; }
.num { text-align: right; }
.total td { font-weight: bold; border-bottom: none; }
.legal { margin-top: 24pt; font-size: 8pt; color: #6b7280; }
.paid { margin-top: 12pt; color: #047857; font-weight: bold; }
"""


@dataclass(frozen=True, slots=True)
class Seller:
    name: str
    address: str = ""
    siren: str = ""
    vat: str = ""
    iban: str = ""
    email: str = ""


def _euros(value: Any) -> str:
    amount = Decimal(str(value or 0)).quantize(Decimal("0.01"))
    whole, _, cents = f"{amount:,.2f}".partition(".")
    return f"{whole.replace(',', ' ')},{cents} €"


def _date(value: Any) -> str:
    if not value:
        return "—"
    return date.fromisoformat(str(value)[:10]).strftime("%d/%m/%Y")


def _lines(text: str | None) -> str:
    return "<br/>".join(escape(line) for line in (text or "").splitlines() if line)


def _html(invoice: dict[str, Any], seller: Seller) -> str:
    client = invoice.get("organization") or {}
    vat_rate = Decimal(str(invoice.get("vat_rate") or 0))
    seller_ids = " · ".join(
        part
        for part in (
            f"SIREN {escape(seller.siren)}" if seller.siren else "",
            f"TVA {escape(seller.vat)}" if seller.vat else "",
        )
        if part
    )
    reverse_charge = (
        "<p>Autoliquidation : TVA due par le preneur (art. 283-2 du CGI, "
        "art. 196 de la directive 2006/112/CE).</p>"
        if vat_rate == 0 and client.get("vat_number")
        else ""
    )
    paid = (
        f'<p class="paid">Acquittée le {_date(invoice.get("paid_on"))}</p>'
        if invoice.get("status") == "paid"
        else ""
    )
    iban = (
        f"<p>Règlement par virement : IBAN {escape(seller.iban)}</p>"
        if seller.iban
        else ""
    )
    return f"""
<h1>Facture {escape(str(invoice.get("number") or ""))}</h1>
<p class="muted">Émise le {_date(invoice.get("issued_on"))}
 · échéance le {_date(invoice.get("due_on"))}</p>

<div class="parties">
  <div class="party"><b>{escape(seller.name)}</b>{_lines(seller.address)}
    {"<br/>" + seller_ids if seller_ids else ""}
    {"<br/>" + escape(seller.email) if seller.email else ""}</div>
  <div class="party"><b>Facturé à : {escape(str(client.get("name") or ""))}</b>
    {_lines(client.get("billing_address"))}
    {"<br/>TVA " + escape(client["vat_number"]) if client.get("vat_number") else ""}
    {"<br/>" + escape(client["billing_email"]) if client.get("billing_email") else ""}
  </div>
</div>

<table>
  <tr><th>Désignation</th><th class="num">Montant HT</th></tr>
  <tr><td>{escape(str(invoice.get("label") or ""))}</td>
      <td class="num">{_euros(invoice.get("amount_eur"))}</td></tr>
  <tr><td>TVA {vat_rate.normalize():f} %</td>
      <td class="num">{_euros(invoice.get("tax_eur"))}</td></tr>
  <tr class="total"><td>Total TTC</td>
      <td class="num">{_euros(invoice.get("total_eur"))}</td></tr>
</table>
{paid}

<div class="legal">
  {iban}
  {reverse_charge}
  <p>Pas d'escompte pour paiement anticipé. En cas de retard de paiement,
  pénalités au taux de trois fois le taux d'intérêt légal et indemnité
  forfaitaire pour frais de recouvrement de 40 € (art. L441-10 du Code de
  commerce).</p>
</div>
"""


def render_invoice_pdf(invoice: dict[str, Any], seller: Seller) -> bytes:
    story = pymupdf.Story(html=_html(invoice, seller), user_css=CSS)
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (56, 56, -56, -56)
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()

    document = pymupdf.open("pdf", buffer.getvalue())
    document.set_metadata(
        {
            "title": f"Facture {invoice.get('number') or ''}",
            "author": seller.name,
            "creator": "Themiros",
        }
    )
    return document.tobytes(garbage=3, deflate=True)
