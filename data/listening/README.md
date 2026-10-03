# Listas de escucha

Lo que el instrumento no puede decidir solo. Un TSV por veredicto, con una columna `veredicto`
vacía y todo lo necesario para juzgar cada clip al lado.

**Esto es lo único no recomputable del repositorio.** Todo lo demás se regenera corriendo el
pipeline; estas etiquetas se pierden para siempre. Se commitean.

## Cómo se llena

Se edita la columna `veredicto` —la última— con **uno de los valores de la tabla de su tanda**, y
nada más. Vocabulario cerrado: sin esto la mitad de las filas dice "ruido" y la otra "noise, poco",
y no se puede contar nada.

Regenerar la lista **preserva lo ya escrito**: se busca por `generation_id`, así que se puede
parar, seguir semanas después y editarla en una planilla. Lo único que la rompe es meter un tab o
un salto de línea dentro de una celda.

```bash
python -m rtb.metrics.listening \
  --manifest data/manifests/xtts_v2.jsonl \
  --manifest data/manifests/qwen3_tts.jsonl \
  --manifest data/manifests/chatterbox_v3.jsonl \
  --verdict trailing_audio
```

**Las filas vienen ordenadas de peor a mejor**, a propósito: si la escucha se corta a la mitad, lo
escuchado es lo que más informa. No reordenar.

**Las tandas no son ciegas: la columna `modelo` está a la vista.** Para las preguntas descriptivas
—¿hay habla o ruido?— pesa poco. Para el veredicto que declara un falso positivo sí puede sesgar,
porque saber de qué modelo es invita a confirmar lo que ya se espera de él. Está así porque el
orden de peor a mejor y el anonimato no conviven; si alguna tanda necesita ser ciega, hay que
exportarla distinto. Los juicios subjetivos de SPEC 7.6 —voseo, prosodia, naturalidad— sí exigen
orden aleatorio, modelo oculto y clips reales mezclados, y ésos **no** se hacen con esta
herramienta.

## `trailing_audio.tsv` — qué hay después de la última palabra

La pregunta es **qué contiene la cola**, no si el modelo es bueno. Ya está medido que suena: el
frame más alto supera los −30 dB del pico del propio clip en el 100 % de las filas (A13).

| valor | cuándo |
|---|---|
| `habla` | la cola tiene palabras que el modelo inventó |
| `ruido` | la cola suena pero no son palabras |
| `ambas` | las dos cosas en el mismo clip |
| `nada` | no se escucha nada: la marca es un falso positivo |
| `corte` | la cola es la frase cortándose, no material extra |

## `trailing_disagreement.tsv` — quién se comió el final

Faltan palabras al final de la transcripción y hay sonido justo donde deberían estar. La pregunta
es **quién se comió el final** (A2). La regla no puede contestarla: el ASR que no lo oyó y el
modelo que lo dijo mal se ven igual desde el texto.

| valor | cuándo |
|---|---|
| `dijo_bien` | el modelo dijo el final entero; el ASR no lo transcribió |
| `dijo_mal` | el modelo pronunció mal el final, y el ASR transcribió lo que pudo |
| `no_lo_dijo` | el modelo se cortó: el truncamiento es real |
| `nada` | no se oye nada después de la última palabra: la marca es un falso positivo |

## `boundary_disagreement.tsv` — de quién es la partición de palabra

El WER vio inserciones y el CER pegado las desmiente: el ASR partió una palabra en dos. La
pregunta es **de quién es la partición** (A1). La regla no puede contestarla porque el mismo patrón
lo produce un rehilamiento fuerte y una aproximante.

| valor | cuándo |
|---|---|
| `dijo_bien` | el modelo dijo la palabra entera; el ASR la partió |
| `dijo_mal` | el modelo pronunció algo distinto, y la partición refleja el audio |
| `dialectal` | lo que cambia es la realización rioplatense: rehilamiento o su ausencia |
| `nada` | no se oye diferencia: la marca es un falso positivo |

## Las columnas son evidencia, no parte del juicio

`cola_s` y `nivel_db` ubican dónde empieza la cola y si es audible; `frase` y `transcript` dicen
qué se pidió y qué oyó el ASR; `habla_s`, `relleno_s` y `total_s` son la anatomía del clip
(SPEC 7.5). Sirven para saber **dónde escuchar**. El veredicto sale del audio.
