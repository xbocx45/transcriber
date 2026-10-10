#!/usr/bin/env bash
# Установка сервера расшифровки речи на чистый сервер Ubuntu 22.04 / 24.04.
#
# Запуск (из папки с файлами программы):
#   sudo bash install.sh
# Со своим доменом (запись A домена должна уже смотреть на этот сервер):
#   sudo DOMAIN=stt.vashsait.ru bash install.sh
#
# Повторный запуск обновляет программу и сохраняет ключи, настройки и скачанную модель.
set -euo pipefail

say()  { printf '\n\033[1;33m==> %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mОшибка: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "запустите через sudo: sudo bash install.sh"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SRC/app.py" ] && [ -f "$SRC/index.html" ] || die "рядом с install.sh должны лежать app.py и index.html"
command -v apt-get >/dev/null || die "нужна Ubuntu или Debian"

APP=/opt/transcriber
DATA=/var/lib/transcriber
ENVF=/etc/transcriber.env
CADDY_VER=2.8.4

say "Ставлю системные пакеты (Python, ffmpeg)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q python3 python3-venv python3-pip ffmpeg curl ca-certificates openssl >/dev/null

say "Готовлю папки и пользователя"
id transcriber >/dev/null 2>&1 || useradd --system --home "$DATA" --shell /usr/sbin/nologin transcriber
mkdir -p "$APP" "$DATA"
install -m 0644 "$SRC/app.py" "$SRC/index.html" "$SRC/requirements.txt" "$APP/"
chown -R transcriber:transcriber "$DATA"

say "Ставлю программу распознавания (несколько минут)"
[ -x "$APP/venv/bin/python" ] || python3 -m venv "$APP/venv"
"$APP/venv/bin/pip" install -q --upgrade pip
"$APP/venv/bin/pip" install -q -r "$APP/requirements.txt"

# ---------- настройки и ключи ----------
if [ ! -f "$ENVF" ]; then
  say "Создаю настройки и ключи доступа"
  RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024 + 0.5}' /proc/meminfo)
  if [ "$RAM_GB" -ge 6 ]; then MODEL=large-v3-turbo; else MODEL=small; fi
  HF=""
  if ! curl -s -m 15 -o /dev/null https://huggingface.co; then
    HF="https://hf-mirror.com"   # если основной сайт с моделями недоступен, качаем с зеркала
  fi
  cat > "$ENVF" <<EOF
# Настройки сервера расшифровки. После изменения: sudo systemctl restart transcriber

# Ключи доступа через запятую, в виде имя:ключ. Имя нужно только для статистики.
# Новый ключ проще добавить командой: sudo transcriber-addkey ИМЯ
TOKENS=bot:$(openssl rand -hex 24),crm:$(openssl rand -hex 24)

# Модель: small (быстрее, хуже), medium, large-v3-turbo (лучше, нужно от 6 ГБ памяти)
MODEL=$MODEL
COMPUTE_TYPE=int8
LANGUAGE=ru
DATA_DIR=$DATA
MAX_UPLOAD_MB=500
EOF
  if [ -n "$HF" ]; then echo "HF_ENDPOINT=$HF" >> "$ENVF"; fi
  chmod 640 "$ENVF"
  chgrp transcriber "$ENVF"
fi
# shellcheck disable=SC1090
set -a; . "$ENVF"; set +a

say "Скачиваю модель $MODEL (один раз, около 0.5 до 1.6 ГБ)"
runuser -u transcriber -- env HOME="$DATA" "$APP/venv/bin/python" - <<EOF
from faster_whisper import WhisperModel
WhisperModel("$MODEL", device="cpu", compute_type="${COMPUTE_TYPE:-int8}", download_root="$DATA/models")
print("модель на месте")
EOF

# ---------- команда для новых ключей ----------
cat > /usr/local/bin/transcriber-addkey <<'EOF'
#!/usr/bin/env bash
# Добавляет ключ доступа для новой программы: sudo transcriber-addkey ИМЯ
set -e
[ "$(id -u)" -eq 0 ] || { echo "запустите через sudo"; exit 1; }
NAME="${1:-prog$(date +%s)}"
NAME="${NAME//[^a-zA-Z0-9_-]/}"
KEY=$(openssl rand -hex 24)
sed -i "s/^TOKENS=\(.*\)$/TOKENS=\1,$NAME:$KEY/" /etc/transcriber.env
systemctl restart transcriber
echo "Ключ для $NAME: $KEY"
EOF
chmod 755 /usr/local/bin/transcriber-addkey

# ---------- служба ----------
say "Настраиваю автозапуск"
cat > /etc/systemd/system/transcriber.service <<EOF
[Unit]
Description=Сервер расшифровки речи
After=network-online.target
Wants=network-online.target

[Service]
User=transcriber
Group=transcriber
WorkingDirectory=$APP
EnvironmentFile=$ENVF
Environment=HOME=$DATA
Environment=HF_HUB_OFFLINE=1
ExecStart=$APP/venv/bin/uvicorn app:app --host 127.0.0.1 --port 9000 --workers 1 --timeout-keep-alive 75
Restart=always
RestartSec=5
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=$DATA

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable transcriber >/dev/null 2>&1
systemctl restart transcriber

# ---------- https ----------
say "Настраиваю защищенный адрес (https)"
if ! command -v caddy >/dev/null; then
  case "$(uname -m)" in
    x86_64) ARCH=amd64 ;;
    aarch64|arm64) ARCH=arm64 ;;
    *) die "неизвестный тип процессора $(uname -m)" ;;
  esac
  TMP=$(mktemp -d)
  curl -fsSL -o "$TMP/caddy.tgz" "https://github.com/caddyserver/caddy/releases/download/v${CADDY_VER}/caddy_${CADDY_VER}_linux_${ARCH}.tar.gz"
  tar -xzf "$TMP/caddy.tgz" -C "$TMP" caddy
  install -m 0755 "$TMP/caddy" /usr/local/bin/caddy
  rm -rf "$TMP"
fi
id caddy >/dev/null 2>&1 || useradd --system --home /var/lib/caddy --create-home --shell /usr/sbin/nologin caddy
mkdir -p /etc/caddy

if [ -z "${DOMAIN:-}" ] && [ -f /etc/caddy/transcriber-domain ]; then
  DOMAIN=$(cat /etc/caddy/transcriber-domain)
fi
if [ -z "${DOMAIN:-}" ]; then
  IP=$(curl -4 -s -m 10 https://api.ipify.org || true)
  [ -n "$IP" ] || IP=$(hostname -I | awk '{print $1}')
  DOMAIN="${IP//./-}.sslip.io"   # адрес по IP сервера, работает без покупки домена
fi
echo "$DOMAIN" > /etc/caddy/transcriber-domain

# у каждой программы на сервере свой файл в /etc/caddy/sites, общий файл только подключает их
mkdir -p /etc/caddy/sites
printf 'import /etc/caddy/sites/*.caddy\n' > /etc/caddy/Caddyfile
cat > /etc/caddy/sites/transcriber.caddy <<EOF
$DOMAIN {
	encode gzip
	request_body {
		max_size 520MB
	}
	reverse_proxy 127.0.0.1:9000 {
		transport http {
			read_timeout 20m
			write_timeout 20m
		}
	}
	log {
		output discard
	}
}
EOF

cat > /etc/systemd/system/caddy.service <<'EOF'
[Unit]
Description=Caddy (https)
After=network-online.target
Wants=network-online.target

[Service]
User=caddy
Group=caddy
ExecStart=/usr/local/bin/caddy run --environ --config /etc/caddy/Caddyfile
ExecReload=/usr/local/bin/caddy reload --config /etc/caddy/Caddyfile --force
Restart=always
AmbientCapabilities=CAP_NET_BIND_SERVICE
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
/usr/local/bin/caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null 2>&1 || die "ошибка в настройке https: sudo caddy validate --config /etc/caddy/Caddyfile"
systemctl enable caddy >/dev/null 2>&1
systemctl restart caddy

if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  ufw allow 22/tcp >/dev/null; ufw allow 80/tcp >/dev/null; ufw allow 443/tcp >/dev/null
fi

say "Жду, пока сервер загрузит модель"
for _ in $(seq 1 60); do
  if curl -s -m 5 http://127.0.0.1:9000/health | grep -q '"ok":true'; then break; fi
  sleep 5
done
HEALTH=$(curl -s -m 5 http://127.0.0.1:9000/health || echo "нет ответа")

BOT_KEY=$(grep '^TOKENS=' "$ENVF" | sed 's/^TOKENS=//' | tr ',' '\n' | grep '^bot:' | cut -d: -f2)
CRM_KEY=$(grep '^TOKENS=' "$ENVF" | sed 's/^TOKENS=//' | tr ',' '\n' | grep '^crm:' | cut -d: -f2)

cat <<EOF

=====================================================================
 Сервер расшифровки установлен.

 Адрес:            https://$DOMAIN
 Проверка:         https://$DOMAIN/health   (должно быть "ok": true)
 Состояние сейчас: $HEALTH

 Ключ для бота MAX (переменная WHISPER_TOKEN в Bothost):
   $BOT_KEY
 Запасной ключ для другой программы:
   $CRM_KEY

 В Bothost добавьте переменные:
   WHISPER_URL   = https://$DOMAIN
   WHISPER_TOKEN = ключ для бота MAX (выше)

 Посмотреть ключи еще раз:   sudo cat $ENVF
 Новый ключ для программы:   sudo transcriber-addkey ИМЯ
 Журнал работы:              sudo journalctl -u transcriber -f
=====================================================================
EOF
