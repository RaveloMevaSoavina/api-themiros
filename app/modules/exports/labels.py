"""Libellés des exports, repris des écrans du dashboard (fr, en)."""

from typing import Any

LABELS: dict[str, dict[str, Any]] = {
    "fr": {
        "title": "Rapport d'évaluation",
        "generator": "Themiros",
        "meta": {
            "workspace": "Espace",
            "run": "Run",
            "run_date": "Date du run",
            "engine": "Version du moteur",
            "corpus": "Corpus",
            "corpus_value": "{size} documents · niveau {level} ({scope})",
            "validation": "Validation humaine",
            "validated": "Validé par {name} le {date}",
            "not_validated": "Non validé",
            "unknown_member": "un membre de l'espace",
            "persona": "Rôle de lecture",
            "exported": "Export",
            "exported_value": "{date} par {author}",
        },
        "exploratory": {
            "title": "Run exploratoire : résultats indicatifs",
            "text": (
                "Le corpus contient {count} documents. Les résultats sont "
                "indicatifs et ne doivent pas être utilisés comme base de "
                "décision formelle."
            ),
            "footer": "Run exploratoire : résultats indicatifs",
        },
        "sections": {
            "limits": "Limites du corpus et de l'analyse",
            "overview": "Comparatif général",
            "pillars": "Piliers : scores objectivés (couche A)",
            "criteria": "Critères : notes évaluatives (couche B)",
            "alerts": "Alertes de cohérence (couche C)",
            "synthesis": "Synthèse",
            "actions": "Plan d'action : lecture {persona}",
            "notes": "Notes : références des preuves",
        },
        "layers": {
            "a": (
                "Couche A : scores objectivés sur 100, calculés de façon "
                "déterministe à partir des variables observées dans le corpus."
            ),
            "b": (
                "Couche B : notes évaluatives de 0 à 3, étayées par des "
                "extraits. Un critère non conclu n'est pas noté."
            ),
            "c": (
                "Couche C : écarts entre scores objectivés et notes "
                "évaluatives, à instruire par un membre de l'espace."
            ),
        },
        "limits": {
            "level": (
                "Niveau de masse critique : {level} ({scope}), {size} documents."
            ),
            "not_concluded": "Critères non conclus : {count} sur {criteria}.",
            "confidence": "Confiance corpus moyenne : {confidence} %.",
            "unscored": ("Piliers sans score (aucune variable renseignée) : {names}."),
            "variables": "Variables non renseignées : {count} sur {total}.",
            "ambiguous": "Critères à la notation ambiguë : {count}.",
            "pending_alerts": "Alertes de cohérence à instruire : {count}.",
            "unqualified": (
                "Alertes non qualifiées par l'IA : {count} (gravité et message "
                "par défaut de la règle)."
            ),
            "layer_b_missing": (
                "Les notes évaluatives (couche B) n'ont pas été calculées par "
                "le moteur de ce run."
            ),
            "layer_c_missing": (
                "Les alertes de cohérence (couche C) n'ont pas été calculées "
                "pour ce run."
            ),
            "restitution_missing": (
                "La synthèse et le plan d'action n'ont pas été calculés pour ce run."
            ),
            "restitution": {
                "not_needed": (
                    "Aucun progrès, faiblesse ni zone à vérifier n'a été retenu "
                    "pour ce run."
                ),
                "disabled": (
                    "Synthèse non rédigée : l'IA était désactivée lors du "
                    "calcul. Les constats gardent un libellé factuel, sans "
                    "conclusion ni plan d'action."
                ),
                "failed": (
                    "Synthèse non rédigée : la rédaction par l'IA a échoué lors "
                    "du calcul. Les constats gardent un libellé factuel, sans "
                    "conclusion ni plan d'action."
                ),
            },
            "corpus_changed": (
                "Le corpus a changé depuis ce calcul : des documents ont été "
                "ajoutés ou retirés. Relancez l'analyse pour actualiser les "
                "résultats."
            ),
            "not_validated": "Ce run n'a pas été validé humainement.",
        },
        "columns": {
            "pillar": "Pilier",
            "version": "Version",
            "score": "Score /100",
            "delta": "Écart",
            "evolution": "Évolution",
            "confidence": "Confiance corpus",
            "documents": "Documents",
            "variable": "Variable",
            "code": "Code",
            "state": "État",
            "value": "Valeur",
            "criterion": "Critère",
            "note": "Note /3",
            "status": "Statut",
            "sub_questions": "Sous-questions documentées",
            "justification": "Justification",
            "ambiguous": "Ambiguë",
            "severity": "Gravité",
            "rule": "Règle",
            "type": "Type",
            "scope": "Objet",
            "message": "Message",
            "qualified": "Qualifiée par l'IA",
            "comment": "Commentaire d'instruction",
            "section": "Rubrique",
            "position": "Rang",
            "text": "Texte",
            "origin": "Origine",
            "action": "Action",
            "reason": "Raison",
            "persona": "Rôle de lecture",
            "notes": "Preuves",
            "number": "N°",
            "document": "Document",
            "page": "Page",
            "excerpt": "Extrait",
            "subject": "Élément évalué",
            "field": "Champ",
            "layer": "Couche",
            "element": "Élément",
            "detail": "Détail",
        },
        "sheets": {
            "summary": "Résumé",
            "overview": "Comparatif",
            "variables": "Variables",
            "criteria": "Critères",
            "alerts": "Alertes",
            "synthesis": "Synthèse",
            "actions": "Plan d'action",
            "notes": "Preuves",
        },
        "layer_codes": {
            "run": "Run",
            "limits": "Limites",
            "a": "A",
            "b": "B",
            "c": "C",
            "synthesis": "Synthèse",
            "actions": "Plan d'action",
            "notes": "Preuves",
        },
        "levels": {
            "exploratoire": "exploratoire",
            "standard": "standard",
            "defendable": "défendable",
        },
        "scope": {
            "exploratoire": "résultats indicatifs",
            "standard": "usage professionnel",
            "defendable": "usage bailleur / audit",
        },
        "state": {
            "present": "Présent",
            "partiel": "Partiel",
            "absent": "Absent",
            "non_renseigne": "Non renseigné",
        },
        "status": {"conclu": "Conclu", "non_conclu": "Non conclu"},
        "evolution": {
            "renforce": "Renforcé",
            "maintenu": "Maintenu",
            "affaibli": "Affaibli",
        },
        "severity": {"mineure": "Mineure", "majeure": "Majeure"},
        "alert_status": {"a_instruire": "À instruire", "instruite": "Instruite"},
        "synthesis_sections": {
            "progres": "Progrès principaux",
            "faiblesse": "Faiblesses principales",
            "a_verifier": "Zones à vérifier",
        },
        "synthesis_empty": {
            "progres": "Aucun pilier renforcé depuis la version précédente.",
            "faiblesse": "Aucune faiblesse retenue.",
            "a_verifier": "Aucune alerte de cohérence à vérifier.",
        },
        "origin": {"pillar": "Pilier", "criterion": "Critère", "alert": "Alerte"},
        "personas": {
            "decideur": "Décideur",
            "pmu": "Chef de projet / PMU",
            "analyste": "Analyste",
        },
        "actions_empty": {
            "completed": (
                "Aucune action : l'IA n'a proposé aucune action étayée pour ce run."
            ),
            "not_needed": (
                "Aucune action : aucune faiblesse, alerte ni critère non conclu "
                "n'a été retenu."
            ),
            "disabled": (
                "Plan d'action non généré : l'IA était désactivée lors du calcul."
            ),
            "failed": (
                "Plan d'action non généré : la rédaction par l'IA a échoué lors "
                "du calcul."
            ),
            "missing": "Plan d'action non calculé pour ce run.",
        },
        "because": "Recommandée parce que {reason}.",
        "conclusion": "Conclusion",
        "no_conclusion": "Conclusion non rédigée pour ce run.",
        "no_scores": "Aucun score de couche A pour ce run.",
        "no_variables": "Aucune variable calculée pour ce pilier.",
        "no_criteria": "Aucune note de couche B pour ce run.",
        "no_alerts": "Aucune alerte de cohérence pour ce run.",
        "no_notes": "Aucune preuve n'est citée dans ce rapport.",
        "no_limits": "Aucune autre limite relevée.",
        "notes_intro": (
            "Preuves du run du {date} (moteur {engine}), une note par page "
            "citée. Les numéros entre crochets renvoient à ces notes."
        ),
        "average": "Score moyen des piliers",
        "not_computed": "Non calculé",
        "yes": "Oui",
        "no": "Non",
        "empty": "—",
        "page": "p. {page}",
        "page_footer": "Page {page}/{total}",
        "extracts": "{shown} extraits cités sur les {count} de cette page.",
    },
    "en": {
        "title": "Evaluation report",
        "generator": "Themiros",
        "meta": {
            "workspace": "Workspace",
            "run": "Run",
            "run_date": "Run date",
            "engine": "Engine version",
            "corpus": "Corpus",
            "corpus_value": "{size} documents · {level} level ({scope})",
            "validation": "Human validation",
            "validated": "Validated by {name} on {date}",
            "not_validated": "Not validated",
            "unknown_member": "a workspace member",
            "persona": "Reading role",
            "exported": "Export",
            "exported_value": "{date} by {author}",
        },
        "exploratory": {
            "title": "Exploratory run: indicative results",
            "text": (
                "The corpus contains {count} documents. The results are "
                "indicative and must not be used as a basis for a formal "
                "decision."
            ),
            "footer": "Exploratory run: indicative results",
        },
        "sections": {
            "limits": "Corpus and analysis limits",
            "overview": "Overall comparison",
            "pillars": "Pillars: objective scores (layer A)",
            "criteria": "Criteria: evaluative ratings (layer B)",
            "alerts": "Consistency alerts (layer C)",
            "synthesis": "Synthesis",
            "actions": "Action plan: {persona} reading",
            "notes": "Notes: evidence references",
        },
        "layers": {
            "a": (
                "Layer A: objective scores out of 100, computed "
                "deterministically from the variables observed in the corpus."
            ),
            "b": (
                "Layer B: evaluative ratings from 0 to 3, backed by excerpts. "
                "A criterion that is not concluded is not rated."
            ),
            "c": (
                "Layer C: gaps between objective scores and evaluative "
                "ratings, to be reviewed by a workspace member."
            ),
        },
        "limits": {
            "level": "Critical mass level: {level} ({scope}), {size} documents.",
            "not_concluded": "Criteria not concluded: {count} of {criteria}.",
            "confidence": "Average corpus confidence: {confidence}%.",
            "unscored": "Pillars without a score (no documented variable): {names}.",
            "variables": "Undocumented variables: {count} of {total}.",
            "ambiguous": "Criteria with an ambiguous rating: {count}.",
            "pending_alerts": "Consistency alerts to review: {count}.",
            "unqualified": (
                "Alerts not qualified by AI: {count} (the rule's default "
                "severity and message)."
            ),
            "layer_b_missing": (
                "Evaluative ratings (layer B) were not computed by this run's engine."
            ),
            "layer_c_missing": (
                "Consistency alerts (layer C) were not computed for this run."
            ),
            "restitution_missing": (
                "The synthesis and action plan were not computed for this run."
            ),
            "restitution": {
                "not_needed": (
                    "No progress, weakness or area to verify was retained for this run."
                ),
                "disabled": (
                    "Synthesis not written: AI was disabled during the "
                    "computation. Findings keep a factual label, with no "
                    "conclusion or action plan."
                ),
                "failed": (
                    "Synthesis not written: AI drafting failed during the "
                    "computation. Findings keep a factual label, with no "
                    "conclusion or action plan."
                ),
            },
            "corpus_changed": (
                "The corpus has changed since this computation: documents were "
                "added or removed. Run the analysis again to refresh the "
                "results."
            ),
            "not_validated": "This run has not been validated by a human.",
        },
        "columns": {
            "pillar": "Pillar",
            "version": "Version",
            "score": "Score /100",
            "delta": "Change",
            "evolution": "Trend",
            "confidence": "Corpus confidence",
            "documents": "Documents",
            "variable": "Variable",
            "code": "Code",
            "state": "State",
            "value": "Value",
            "criterion": "Criterion",
            "note": "Rating /3",
            "status": "Status",
            "sub_questions": "Documented sub-questions",
            "justification": "Justification",
            "ambiguous": "Ambiguous",
            "severity": "Severity",
            "rule": "Rule",
            "type": "Type",
            "scope": "Subject",
            "message": "Message",
            "qualified": "Qualified by AI",
            "comment": "Review comment",
            "section": "Section",
            "position": "Rank",
            "text": "Text",
            "origin": "Origin",
            "action": "Action",
            "reason": "Reason",
            "persona": "Reading role",
            "notes": "Evidence",
            "number": "No.",
            "document": "Document",
            "page": "Page",
            "excerpt": "Excerpt",
            "subject": "Assessed item",
            "field": "Field",
            "layer": "Layer",
            "element": "Item",
            "detail": "Detail",
        },
        "sheets": {
            "summary": "Summary",
            "overview": "Comparison",
            "variables": "Variables",
            "criteria": "Criteria",
            "alerts": "Alerts",
            "synthesis": "Synthesis",
            "actions": "Action plan",
            "notes": "Evidence",
        },
        "layer_codes": {
            "run": "Run",
            "limits": "Limits",
            "a": "A",
            "b": "B",
            "c": "C",
            "synthesis": "Synthesis",
            "actions": "Action plan",
            "notes": "Evidence",
        },
        "levels": {
            "exploratoire": "exploratory",
            "standard": "standard",
            "defendable": "defensible",
        },
        "scope": {
            "exploratoire": "indicative results",
            "standard": "professional use",
            "defendable": "donor / audit use",
        },
        "state": {
            "present": "Present",
            "partiel": "Partial",
            "absent": "Absent",
            "non_renseigne": "Not documented",
        },
        "status": {"conclu": "Concluded", "non_conclu": "Not concluded"},
        "evolution": {
            "renforce": "Strengthened",
            "maintenu": "Maintained",
            "affaibli": "Weakened",
        },
        "severity": {"mineure": "Minor", "majeure": "Major"},
        "alert_status": {"a_instruire": "To review", "instruite": "Reviewed"},
        "synthesis_sections": {
            "progres": "Main progress",
            "faiblesse": "Main weaknesses",
            "a_verifier": "Areas to verify",
        },
        "synthesis_empty": {
            "progres": "No pillar strengthened since the previous version.",
            "faiblesse": "No weakness retained.",
            "a_verifier": "No consistency alert to verify.",
        },
        "origin": {"pillar": "Pillar", "criterion": "Criterion", "alert": "Alert"},
        "personas": {
            "decideur": "Decision-maker",
            "pmu": "Project manager / PMU",
            "analyste": "Analyst",
        },
        "actions_empty": {
            "completed": "No action: AI proposed no supported action for this run.",
            "not_needed": (
                "No action: no weakness, alert or unconcluded criterion was retained."
            ),
            "disabled": (
                "Action plan not generated: AI was disabled during the computation."
            ),
            "failed": (
                "Action plan not generated: AI drafting failed during the computation."
            ),
            "missing": "Action plan not computed for this run.",
        },
        "because": "Recommended because {reason}.",
        "conclusion": "Conclusion",
        "no_conclusion": "No conclusion was written for this run.",
        "no_scores": "No layer A score for this run.",
        "no_variables": "No variable computed for this pillar.",
        "no_criteria": "No layer B rating for this run.",
        "no_alerts": "No consistency alert for this run.",
        "no_notes": "No evidence is cited in this report.",
        "no_limits": "No other limit identified.",
        "notes_intro": (
            "Evidence from the run of {date} (engine {engine}), one note per "
            "cited page. Numbers in brackets refer to these notes."
        ),
        "average": "Average pillar score",
        "not_computed": "Not computed",
        "yes": "Yes",
        "no": "No",
        "empty": "—",
        "page": "p. {page}",
        "page_footer": "Page {page}/{total}",
        "extracts": "{shown} of the {count} excerpts from this page are quoted.",
    },
}


def labels_for(locale: str) -> dict[str, Any]:
    return LABELS.get(locale, LABELS["en"])
