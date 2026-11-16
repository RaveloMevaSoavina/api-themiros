# Jeux d'évaluation de la recherche des extraits

Ces jeux servent à mesurer si la recherche des extraits (couche B) trouve les
bons passages des documents d'un projet. Chaque réglage de la recherche
(seuils, recherche hybride, taille des segments) se
mesure sur eux avant d'être déployé.

## Contenu d'un jeu

Un fichier JSON par projet, sur le modèle de [modele.json](modele.json) :

- `documents` : identifiants des documents du projet, déjà indexés ;
- `questions` : les sous-questions à tester, avec leur critère, leur texte et
  leurs preuves attendues, recopiés du cadre d'évaluation ;
- `relevant` : pour chaque sous-question, les passages qui y répondent
  vraiment, chacun donné par l'identifiant de son document et une **citation
  courte** (une phrase, au moins 12 caractères) copiée du texte. Liste vide
  si les documents ne répondent pas à la sous-question : le moteur ne devrait
  alors rien retenir.

La citation est comparée sans tenir compte de la casse ni des espaces. Elle
doit tenir dans un seul segment : une phrase suffit.

Cible de départ : 3 projets réels, une vingtaine de sous-questions chacun,
dont quelques-unes sans réponse dans les documents.

## Annoter

```bash
python -m app.modules.evals.retrieval annotate evals/retrieval/projet-x.json -o annotation.md
```

`annotation.md` liste, pour chaque sous-question, les 20 meilleurs candidats
de la recherche, sans seuil, avec leur document. Les passages déjà annotés
sont marqués ✓. Pour chaque passage qui répond à la question, copier une
phrase dans `relevant`. Si un bon passage n'apparaît pas parmi les candidats,
l'ajouter quand même, depuis le document : c'est justement ce que la
recherche manque.

## Mesurer

```bash
python -m app.modules.evals.retrieval run evals/retrieval/projet-x.json
python -m app.modules.evals.retrieval run evals/retrieval/projet-x.json \
    --min-similarity 0.35 --max-per-document 2 --json resultat.json
```

Le banc compare la recherche actuelle à celle de la version 1.3.0 (vecteurs
seuls) sur ce que reçoit l'IA, c'est-à-dire les premiers extraits de chaque
sous-question :

| Mesure | Ce qu'elle dit |
|---|---|
| Rappel | Part des passages annotés que l'IA reçoit |
| Sous-questions trouvées | Part des sous-questions dont l'IA reçoit au moins un bon passage |
| Précision | Part des extraits envoyés qui sont pertinents |
| MRR | Le premier bon passage arrive-t-il en tête (1 = toujours premier) |
| Sans réponse laissées vides | Part des sous-questions sans réponse pour lesquelles rien n'est envoyé |

Les passages annotés que l'IA ne reçoit pas sont listés à la fin : ce sont
les cas à examiner.

Le banc a besoin des mêmes variables que le worker (`SUPABASE_URL`,
`SUPABASE_SERVICE_ROLE_KEY`, `OPENAI_API_KEY`). Il ne crée pas de run et
n'appelle pas l'IA de notation : seuls les vecteurs des requêtes sont
calculés, pour quelques centimes.

Les jeux contiennent des extraits de documents de projets : ne les versionner
que si les documents peuvent l'être.
