# Déploiement Docker de l'API Themiros

La pile contient l'API FastAPI et Caddy. Caddy publie automatiquement l'API en
HTTPS et renouvelle son certificat. Le port applicatif 8000 reste uniquement
accessible depuis le serveur.

## Prérequis serveur

- Docker Engine avec le plugin Docker Compose ;
- les ports entrants TCP 80 et 443, ainsi que UDP 443, ouverts ;
- un enregistrement DNS `A` ou `AAAA` pour le sous-domaine de l'API pointant
  vers le serveur ;
- aucun autre service ne doit utiliser les ports 80 ou 443.

## Premier déploiement

Copier ou cloner le dossier `api-themiros` sur le serveur, puis :

```bash
cd api-themiros
cp .env.production.example .env.production
chmod 600 .env.production
```

Renseigner toutes les valeurs de `.env.production`. `API_DOMAIN` doit contenir
uniquement le nom DNS, par exemple `api.example.com`. Dans
`CORS_ALLOWED_ORIGINS`, indiquer l'origine exacte du dashboard, avec `https://`
mais sans chemin.

Déployer ensuite :

```bash
./deploy/deploy.sh
```

Vérifier les deux sondes :

```bash
curl https://api.example.com/health
curl https://api.example.com/ready
```

`/health` vérifie le processus API. `/ready` vérifie également la connexion à
Supabase et doit répondre avec le statut `ready`.

## Configuration du dashboard

Le dashboard doit être construit avec l'URL publique de l'API :

```env
VITE_API_URL=https://api.example.com
```

Les variables Vite sont injectées à la compilation : reconstruire et redéployer
le dashboard après cette modification.

## Mise à jour

Après avoir récupéré la nouvelle version du code :

```bash
git pull
./deploy/deploy.sh
```

Pour suivre les journaux :

```bash
docker compose --env-file .env.production logs -f --tail=100
```

Pour connaître l'état des services :

```bash
docker compose --env-file .env.production ps
```

## Arrêt et sauvegarde

Arrêter les conteneurs sans supprimer les certificats :

```bash
docker compose --env-file .env.production down
```

Ne pas ajouter l'option `--volumes` : le volume `caddy_data` contient les
certificats TLS. Sauvegarder `.env.production` dans un gestionnaire de secrets,
et non dans Git.

