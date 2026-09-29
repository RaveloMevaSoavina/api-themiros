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
```

Arrêter l'API :

```bash
docker compose --env-file .env.production down
```
