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

## Los comandos

```bash
python -m rtb.metrics.corrections --category normalizacion --model sintetica \
  --manifest data/manifests/sintetica.jsonl

uv run scripts/stage_listening_clips.py --verdict normalizacion \
  --listening data/corrections --dest audio/corrections --column revisado
```

El segundo junta los wavs en `audio/corrections/normalizacion/`, numerados por la fila del TSV, y
con `--pending-only` trae sólo los que faltan. Esa carpeta no se commitea: el artefacto es el TSV.

Regenerar **preserva lo ya transcripto**: se busca por `audio_sha`, así que se puede parar, seguir
otro día y editarlo en una planilla. Lo único que lo rompe es meter un tab o un salto de línea
dentro de una celda.
