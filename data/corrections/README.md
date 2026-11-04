# Transcripciones a oído

Lo que el modelo **efectivamente dijo**, escrito por una persona, para las filas donde el ASR no
decide nada. Hoy es sólo `normalizacion`.

**Esto es lo único no recomputable del repositorio, junto con `data/listening/`.** Se commitea.

## Por qué existe

En `normalizacion` Whisper se equivoca en las dos direcciones y a veces en el mismo clip: inventa
errores que el modelo no cometió y repara los que sí. En `nor_005` escribió el teléfono correcto
que el modelo nunca dijo, y de paso puso `escribia` donde iba `o escribi a`. Ninguna regla sobre el
transcript separa eso, porque la información no está en el texto (A6).

Con la transcripción a oído la comparación queda entre dos textos en palabras y el ASR sale del
medio. Y el conjunto sirve para lo otro: medir el WER de Whisper sobre habla sintética
rioplatense, y comparar otro ASR sin regenerar una sola síntesis.

## Cómo se llena

La columna `dicho` viene **pre-cargada con el transcript de Whisper**, que acierta casi toda la
prosa. El trabajo es escuchar, corregir lo que falla y marcar `revisado` con cualquier cosa.

- **Los números van en palabras.** `uno uno cuatro cinco` y no `1-1-4-5`: la referencia contra la
  que se compara está en palabras. Una fila marcada que todavía tenga dígitos la señala la
  herramienta al regenerar, y la rechaza un test.
- **Escribir lo que se oye, no lo que debería haber dicho.** Si el modelo se come un dígito, va sin
  ese dígito.
- **Sin marcar no cuenta.** `dicho` pre-cargado y `revisado` vacío significa que nadie lo escuchó.
- **`[ininteligible]` para el habla que no se puede escribir.** Cuenta como una palabra y marca que
  ahí hay sonido que no se deja transcribir; el largo real no lo dice la transcripción sino la
  cola. Es para el balbuceo, no para una palabra dudosa: ante la duda va lo que suena.

**La forma hablada esperada no está en la planilla, a propósito.** Ni `verbalizacion` ni
`alternativas`. A la vista, se escribe lo que el modelo debería haber dicho en vez de lo que dijo,
y el conjunto queda arruinado sin que se note después. Un test lo mantiene afuera. La columna
`frase` sí está: es el texto escrito que recibió el modelo, con los dígitos sin expandir, y ubica
el clip sin decir cómo había que leerlo.

## Una fila por audio distinto

La clave es `audio_sha`, no el `generation_id`: la transcripción describe **ese audio**. Las
semillas que producen bytes idénticos comparten fila, y `n` dice cuántas filas del grid cubre cada
una. Sintética es determinista, así que sus 20 clips cubren las 60 filas; los otros cinco modelos
dan audio distinto por semilla y no tienen ese atajo.

Si un clip se regenera y cambia, su `audio_sha` cambia y la fila es nueva: la transcripción vieja
ya no describe ese audio y hay que rehacerla. Eso es a propósito.

**El archivo acumula modelos, así que hay que pasarlos todos.** Reconstruir con menos `--model` de
los que el TSV ya tiene borraría transcripciones hechas, y la herramienta aborta antes de
escribir en vez de dejarte el archivo mutilado.

## Sondear un modelo sin transcribirlo entero

`--one-seed MODELO` deja un audio por cada (hablante, frase), para ese modelo solo. Sirve para
preguntar si el ASR retrata bien a un modelo antes de comprometer sus 60 clips: 20 filas, diez
minutos. Va por modelo y no global, así uno sondeado convive en el mismo TSV con otros
transcriptos enteros, y hay que repetirlo en cada corrida o el modelo sondeado se expande a 60.

El descarte ocurre **después** de agrupar por hash: si las tres semillas son el mismo audio, la
celda ya era una sola fila y `n` sigue diciendo 3.

## Los comandos

```bash
python -m rtb.metrics.corrections --category normalizacion \
  --one-seed pocket_tts_es \
  --model sintetica     --manifest data/manifests/sintetica.jsonl \
  --model pocket_tts_es --manifest data/manifests/pocket_tts_es.jsonl \
  --model xtts_v2       --manifest data/manifests/xtts_v2.jsonl \
  --model chatterbox_v3 --manifest data/manifests/chatterbox_v3.jsonl \
  --model qwen3_tts     --manifest data/manifests/qwen3_tts.jsonl

uv run scripts/stage_listening_clips.py --verdict normalizacion \
  --listening data/corrections --dest audio/corrections --column revisado
```

El segundo junta los wavs en `audio/corrections/normalizacion/`, numerados por la fila del TSV, y
con `--pending-only` —que conviene siempre— trae sólo los que faltan; los huecos en la numeración
dicen cuántos ya están. Esa carpeta no se commitea: el artefacto es el TSV.

Regenerar **preserva lo ya transcripto**: se busca por `audio_sha`, así que se puede parar, seguir
otro día y editarlo en una planilla. Lo único que lo rompe es meter un tab o un salto de línea
dentro de una celda.
