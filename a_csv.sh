#!/usr/bin/env bash
# Uso: ./a_csv.sh entrada.txt [salida.csv]

entrada="$1"
salida="${2:-${entrada%.*}.csv}"

if [[ -z "$entrada" || ! -f "$entrada" ]]; then
    echo "Uso: $0 fichero_entrada [fichero_salida.csv]" >&2
    exit 1
fi

{
    echo "image,score"
    tr -d '\r' < "$entrada" | awk 'NF >= 2 { print $1 "," $2 }'
} > "$salida"

echo "Creado: $salida ($(($(wc -l < "$salida") - 1)) filas)"
