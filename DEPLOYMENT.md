# Déploiement Docker de l'API Themiros

L'API fonctionne dans Docker sur `127.0.0.1:8010`. Le Caddy déjà installé sur
le serveur publie `api.evoranq.com` en HTTPS. Le port 8010 n'est jamais exposé
sur Internet et le port 8000 reste réservé à Supabase Envoy.

## Architecture du serveur

- `supabase.evoranq.com` : Supabase Envoy sur le port 8000 ;
- `api.evoranq.com` : Caddy vers `127.0.0.1:8010` ;
- `app.themiros.com` : dashboard hébergé sur Vercel.

Le DNS de `api.evoranq.com` doit pointer vers l'adresse IP du serveur.

## Premier déploiement

Cloner le dépôt sur le serveur :

```bash
git clone https://github.com/RaveloMevaSoavina/api-themiros.git
cd api-themiros
cp .env.production.example .env.production
chmod 600 .env.production
nano .env.production
```

Renseigner dans `.env.production` les nouvelles clés OpenAI et Supabase. Ne
jamais placer de secret dans `.env.production.example`. Vérifier notamment :

```env
API_DOMAIN=api.evoranq.com
API_HOST_PORT=8010
APP_ENV=production
APP_DEBUG=false
CORS_ALLOWED_ORIGINS=https://app.themiros.com
```

Construire et démarrer l'API :

```bash
./deploy/deploy.sh
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/ready
```

## Raccorder le Caddy existant

Ne pas remplacer le fichier Caddy existant : il contient déjà la configuration
de Supabase et probablement celle de n8n. Ajouter le contenu de
`deploy/Caddyfile.api` à `/etc/caddy/Caddyfile` :

```bash
sudo nano /etc/caddy/Caddyfile
```

Valider la configuration avant tout rechargement :

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
sudo systemctl status caddy --no-pager
```

Effectuer ensuite les tests publics :

```bash
curl https://api.evoranq.com/health
curl https://api.evoranq.com/ready
```

`/health` vérifie le processus API. `/ready` vérifie aussi l'accès à Supabase et
doit répondre avec le statut `ready`.

## Configuration du dashboard

Configurer l'environnement de production du dashboard/Vercel, puis lancer un
nouveau déploiement du frontend :

```env
VITE_API_URL=https://api.evoranq.com
```

## Facturation Stripe

La page Facturation du dashboard (administrateurs de l'organisation) lit
l'offre, le quota et les factures en base ; la souscription en ligne passe par
Stripe Checkout, la gestion du paiement par le portail client Stripe, et le
webhook recopie abonnements et factures (migration
`20261124000001_stripe_billing.sql`).

1. **Clés** dans `.env.production` : `STRIPE_SECRET_KEY` (clé secrète ou
   restreinte), `DASHBOARD_URL=https://app.themiros.com`, mentions
   `BILLING_SELLER_*` des factures manuelles. Redémarrer l'API.
2. **Produits et prix** : `docker compose --env-file .env.production exec api
   python -m app.modules.billing.bootstrap` (idempotent ; crée les prix
   annuels `themiros_essential_annual` et `themiros_professional_annual`).
3. **Tableau de bord Stripe** :
   - Stripe Tax activé, enregistrement TVA France ; origine = siège social.
   - Paramètres › Factures : préfixe `THS` (la série `TH-AAAA-NNNN` reste aux
     factures manuelles du backoffice).
   - Moyens de paiement : carte et prélèvement SEPA.
   - Portail client : mise à jour du moyen de paiement, historique des
     factures, adresse et numéro de TVA, résiliation **en fin de période** ;
     changement d'offre désactivé (il passe par l'équipe).
4. **Webhook** : point d'entrée `https://api.evoranq.com/api/v1/billing/webhook`,
   même version d'API que `STRIPE_API_VERSION`, événements
   `checkout.session.completed`, `customer.subscription.created|updated|deleted`,
   `invoice.finalized|paid|payment_failed|voided|marked_uncollectible`,
   `customer.updated`. Copier le secret de signature dans
   `STRIPE_WEBHOOK_SECRET` et redémarrer l'API.
5. **Dashboard** : `VITE_BILLING_ENABLED=true`, puis redéployer le frontend.
   Backoffice : `VITE_STRIPE_DASHBOARD_URL=https://dashboard.stripe.com`.

En local : clés de test, puis
`stripe listen --forward-to localhost:8000/api/v1/billing/webhook` (le secret
`whsec_…` affiché va dans `STRIPE_WEBHOOK_SECRET`). Carte `4242 4242 4242 4242`
pour un paiement réussi, `4000 0000 0000 0341` pour un échec au
renouvellement ; les horloges de test Stripe simulent le renouvellement annuel.

Les événements reçus sont journalisés dans `stripe_events` : `error` non vide
= échec, que Stripe renvoie pendant trois jours.

## Worker d'ingestion documentaire

`deploy/deploy.sh` démarre aussi le service `ingestion-worker`, construit à
partir de la même image que l'API. Il n'expose aucun port : il lit la file
`ingestion_jobs` dans Supabase, télécharge les fichiers du bucket `documents`
et écrit les résultats avec la clé `service_role`.

Prérequis côté Supabase, dans cet ordre (dépôt `dashboard-themiros`) :

1. `20261006000001_document_ingestion.sql`
2. `20261006000002_reconcile_documents_schema.sql`
3. `20261006000003_document_deletion.sql`

L'image embarque Tesseract (fra, eng, por, spa), le modèle spaCy
`xx_ent_wiki_sm` et le modèle d'embeddings int8 sous `/opt/fastembed` : le
premier build prend plusieurs minutes et l'image dépasse 2 Go. Prévoir au moins
4 Go de RAM et 2 cœurs pour l'API et le worker, en plus de Supabase si celui-ci
tourne sur le même serveur.

Réglages utiles dans `.env.production` :

```env
# Documents traités en parallèle par conteneur (1 par tranche de 4 cœurs).
INGESTION_WORKER_CONCURRENCY=1
# Threads ONNX pour les embeddings ; à limiter si Supabase partage le serveur.
INGESTION_EMBEDDING_THREADS=2
INGESTION_OCR_PARALLELISM=2
```

Vérifier le worker :

```bash
docker compose --env-file .env.production ps ingestion-worker
docker compose --env-file .env.production logs -f --tail=100 ingestion-worker
```

Un document chargé depuis le dashboard doit apparaître dans les journaux
(`Ingestion job … completed`). En cas de charge, plusieurs workers peuvent
tourner : la file PostgreSQL répartit les tâches.

```bash
docker compose --env-file .env.production up -d --scale ingestion-worker=2
```

## Mise à jour de l'API

```bash
cd api-themiros
git pull --ff-only
./deploy/deploy.sh
curl https://api.evoranq.com/health
```

Consulter l'état et les journaux :

```bash
docker compose --env-file .env.production ps
docker compose --env-file .env.production logs -f --tail=100 api
docker compose --env-file .env.production logs -f --tail=100 ingestion-worker
```

Arrêter l'API :

```bash
docker compose --env-file .env.production down
```
