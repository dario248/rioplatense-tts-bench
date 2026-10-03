# SPEC — `rioplatense-tts-bench`

Documento de diseño para implementación asistida. Describe **qué** construir y **por qué**;
las decisiones de implementación puntual quedan abiertas salvo donde se indique lo contrario.

> **Nota de idioma:** este documento y los comentarios internos van en español.
> El `README.md` público, los docstrings y todos los identificadores de código van en **inglés**

---

## 1. Objetivo

Benchmark reproducible de **síntesis de voz zero-shot en español rioplatense**.

Los modelos TTS multilingües declaran "soporte de español", pero ninguno reporta qué pasa con
los rasgos del dialecto rioplatense: yeísmo rehilado, vibrante múltiple, voseo, léxico local.
Este proyecto mide eso con métricas objetivas y publica los resultados.

**Pregunta que responde:** en 2026, ¿qué tan lejos está el TTS open source zero-shot de una voz
rioplatense creíble, y cuánto cuesta esa distancia en cómputo?

**Dos ejes, y no se miden igual.** La **calidad dialectal** ordena entre sí a los modelos zero-shot
y deja a Sintética fuera de competencia: un modelo fine-tuneado sobre la voz que sintetiza no
responde la misma pregunta. El **costo de inferencia** los compara a todos, Sintética incluida —
ahí la pregunta es si el modelo entra en el equipo de quien lo quiera correr, y eso no depende de
cómo llegó la voz.

### No-objetivos

- **No hay fine-tuning.** Todo es zero-shot. Sintética (modelo fine-tuned propio) entra **fuera de
  competencia en el eje de calidad** y como fila comparable en el de costo.
- No es una librería de TTS ni un wrapper de inferencia de uso general.
- No es un servicio. No hay API, ni Docker, ni deploy en v0/v1.
- No se persigue SOTA de calidad; se persigue **medición honesta y reproducible**.

---

## 2. Alcance por versión

### v0 — mínimo publicable

| Eje | Alcance |
|---|---|
| Modelos | XTTS-v2, Qwen3-TTS, Chatterbox Multilingual v3, + Sintética fuera de competencia |
| Hablantes | 4 de SLR61 (2M / 2F), los mismos para todos los modelos |
| Frases | 45, en 10 categorías |
| Seeds | 3 por combinación |
| Métricas | WER/CER, similitud de hablante, tasa de fallo, RTF en GPU y en CPU, memoria pico |
| Salida | Tabla de resultados + README con hallazgos |

Total = 45 × 4 × 3 × 3 modelos = **1620 síntesis** zero-shot, más la corrida de Sintética sobre
esos mismos hablantes y el subset de costo en CPU (6.3).

El set de frases de v0 ya trae 20 targets de yeísmo y 8 de rótica por hablante y seed
(28 × 4 × 3 × 3 = **1008 mediciones** de acento), así que el corpus soporta la métrica de la
sección 7.4 aunque su implementación esté planificada para v1.

### v1 — ampliación

- Modelos: + Voxtral TTS, Fish Audio S2, Kokoro (Kokoro sin clonado: sirve de piso de latencia).
- Frases: hasta ~50.
- **Métrica de acento rioplatense** (sección 7.4). Es el diferencial del proyecto.
- Métrica semi-automática de voseo y prosodia (sección 7.6), que en v0 van a escucha humana.
- Métrica perceptual sin referencia (UTMOS / NISQA), reportada con salvedades.
- Dataset de audios publicado en HuggingFace.
- Recuperar `s_implosiva` **solo** si se incorporan referencias de habla espontánea (ver 5.1).

### Fuera de alcance por ahora

Space interactivo de comparación ciega (proyecto separado, consumirá el dataset de este repo).

---

## 3. Principio de arquitectura — leer antes de escribir código

**La frontera entre etapas son archivos en disco, no imports.**

Los tres modelos TTS tienen dependencias mutuamente incompatibles (versiones distintas de
`transformers`, de `torch`, de `numpy`). No se debe intentar hacerlos convivir en un entorno.

```
Etapa 1: GENERACIÓN     un entorno POR MODELO, sucio, desechable
                        ↓ escribe wavs + manifest JSONL
Etapa 2: MÉTRICAS       entornos limpios de sólo-lectura de audio
                        ↓ lee wavs + manifest, escribe metrics JSONL
Etapa 3: ANÁLISIS       stdlib (v0) / pandas + matplotlib
                        ↓ tablas y figuras
```

La etapa 2 arrancó como **un** entorno con whisper y speechbrain juntos y terminó partida en dos,
`eval.txt` y `speaker_sim.txt`, por una razón medida que está escrita en el segundo: speechbrain
resuelve limpio contra el entorno de ASR pero se trae torch con wheels de CUDA 13, y `ctranslate2`
—que no declara sus libs de CUDA como dependencia— linkea cudnn de afuera. Es un conflicto de
biblioteca compartida que el resolvedor no ve y que aparecería recién en la próxima
re-transcripción. Partir no cuesta nada justamente porque la frontera son archivos: el pase de
similitud escribe su JSONL y el reporte lo lee sin torch.

La etapa 3 de v0 no usa pandas: las tablas se renderizan en `rtb.report.tables` con stdlib, para
que el reporte corra en cualquiera de los entornos y en CI.

Consecuencias que hay que respetar:

1. **La etapa 2 nunca importa una librería de TTS.** Ni indirectamente.
2. **`rtb.schema` y `rtb.manifest` no importan nada fuera de stdlib + pydantic.** Son lo único
   que se instala en todos los entornos. Si el core arrastra `torch`, el diseño falló.
3. Las etapas son asincrónicas en el tiempo: se puede generar XTTS hoy, Qwen3 en tres semanas,
   y evaluar todo junto después. Ninguna etapa asume que otra ya corrió.
4. Los adapters de modelo importan sus dependencias pesadas **de forma lazy**, dentro de
   `load()`, nunca a nivel de módulo. Así `rtb.generate.base` es importable sin GPU.

---

## 4. Estructura del repositorio

```
rioplatense-tts-bench/
├── README.md                    # público, en inglés, orientado a resultados
├── SPEC.md                      # este documento: qué construir y por qué
├── ANALYSIS.md                  # preguntas abiertas SOBRE los resultados, con su costo
├── pyproject.toml               # SOLO el core: pydantic + pyyaml. Nada más.
├── configs/
│   ├── phrases.yaml             # frases + categoría + hipótesis de falla + targets
│   ├── speakers.yaml            # 4 hablantes SLR61, clips de referencia y de control
│   ├── grid.yaml                # seeds, ref_clip_indices, warm-up. COMPARTIDO por los modelos
│   ├── thresholds.yaml          # umbrales de deteccion de fallo, versionados
│   └── models/
│       ├── dummy.yaml           # ruido blanco, para validar el runner sin GPU
│       ├── xtts_v2.yaml
│       ├── qwen3_tts.yaml
│       ├── chatterbox_v3.yaml
│       └── sintetica.yaml
├── src/rtb/
│   ├── schema.py                # modelos de datos. SIN deps pesadas.
│   ├── manifest.py              # lectura/escritura JSONL append-only. SIN deps pesadas.
│   ├── config.py                # carga y validación de los YAML
│   ├── textnorm.py              # normalización de texto, versionada. Core: la usan configs y métricas.
│   ├── logconf.py               # logging estructurado
│   ├── generate/
│   │   ├── base.py              # TTSAdapter (ABC) + SynthesisResult
│   │   ├── runner.py            # loop idempotente sobre el grid
│   │   ├── seeding.py           # seed de python/numpy/torch/cuda, con imports lazy
│   │   ├── audio_io.py          # leer/escribir wav, stdlib. Compartido por los adapters
│   │   ├── dummy.py             # ruido blanco, doble de prueba del runner
│   │   ├── xtts.py
│   │   ├── qwen3.py
│   │   ├── chatterbox.py
│   │   └── sintetica.py         # in-process, local. Ver 6.3
│   ├── metrics/
│   │   ├── asr.py               # transcripcion con Whisper. El paso caro, se corre una vez
│   │   ├── transcribe.py        # el pase de audio: escribe transcripts.jsonl y audio_shape.jsonl
│   │   ├── audio_shape.py       # envolvente de energia: separa habla de relleno
│   │   ├── anatomy.py           # el cuarto tramo: separa la cola del habla, uniendo con el ASR
│   │   ├── intelligibility.py   # WER / CER sobre los transcripts. Barato, recalculable
│   │   ├── speaker_sim.py       # coseno ECAPA-TDNN + techo/piso. Entorno propio, en CPU
│   │   ├── latency.py           # agregación de timings del manifest
│   │   ├── ground_truth.py      # filas de audio real desde speakers.yaml (5.5)
│   │   ├── speaking_rate.py     # la vara: s/palabra medido en los clips reales
│   │   ├── duration_screen.py   # triage de duración sin audio. NO es la tasa publicada
│   │   ├── failures.py          # alucinación, truncamiento, loop — necesita transcripción
│   │   ├── listening.py         # listas de escucha: lo que el instrumento no puede decidir solo
│   │   ├── sensitivity.py       # un punto de barrido de umbral, compartido por las dos tasas
│   │   ├── accent.py            # v1 — métrica rioplatense
│   │   └── audio_utils.py       # lector PCM (stdlib) + resample a 16 kHz (torchaudio, lazy)
│   └── report/
│       ├── tables.py
│       └── figures.py
├── envs/                        # un par por modelo, nombrado como el modelo
│   ├── xtts_v2.txt              # la receta: pines minimos, con el motivo de cada uno
│   ├── xtts_v2.lock.txt         # el registro: uv pip freeze de la maquina que genero el manifest
│   ├── qwen3_tts.txt            # qwen-tts. Python 3.12, y CUDA a elegir segun el driver
│   ├── chatterbox_v3.txt
│   ├── eval.txt                 # ASR: whisper sobre ctranslate2, en GPU
│   ├── speaker_sim.txt          # similitud: speechbrain + torch CPU. Aparte, ver seccion 3
│   └── README.md                # cómo levantar cada entorno
├── notebooks/
│   ├── 00_generate_xtts.ipynb
│   ├── 01_generate_qwen3.ipynb
│   ├── 02_generate_chatterbox.ipynb
│   ├── 10_metrics.ipynb
│   └── 20_analysis.ipynb
├── scripts/                     # entrypoints PEP 723 (uv run) para uso local
│   ├── build_reference_clips.py # construye los clips de speakers.yaml desde el corpus
│   └── make_placeholder_refs.py # ruido en esas rutas, para correr sin el corpus
├── tests/                       # tests del core, sin GPU, corren en CI
├── data/
│   ├── manifests/               # JSONL — SE COMMITEAN
│   ├── metrics/                 # JSONL — SE COMMITEAN
│   ├── listening/               # TSV anotado a oído — SE COMMITEA. Lo único NO recomputable
│   ├── samples/                 # ~30 wavs curados — SE COMMITEAN
│   └── refs/                    # clips de referencia construidos — NO se commitean
└── audio/                       # NO se commitea (.gitignore)
```

---

## 5. Modelos de datos

### 5.1 `configs/phrases.yaml`

```yaml
- id: rot_003
  text: "Recién llegó del río, con la ropa mojada."
  category: rotica
  failure_hypothesis: "Frase de doble fenómeno: tres róticas medibles y un target de yeísmo."
  targets:                              # segmentos medibles automáticamente
    - {word: "Recién", phenomenon: rhotic}
    - {word: "llegó",  phenomenon: yeismo}
    - {word: "río",    phenomenon: rhotic}
    - {word: "ropa",   phenomenon: rhotic}
  n_words: 8                            # opcional, funciona como checksum del texto
```

Campos: `id`, `text`, `category`, `failure_hypothesis`, `targets`, `n_words`, y para la categoría
`normalizacion` además `expected_verbalization` y `accepted_alternatives` (ver más abajo).
`notes` es opcional y sirve para advertencias de medición.

Las diez categorías activas: `yeismo_rehilado`, `rotica`, `voseo`, `enunciados_cortos`,
`lexico_local`, `toponimia`, `normalizacion`, `anglicismos`, `prosodia`, `frase_larga`.

#### Targets — regla de marcado

`targets` es una lista de objetos con `word` y `phenomenon` (`yeismo` | `rhotic`), no una lista
de strings: desde que hay dos fenómenos medibles, el fenómeno tiene que viajar con la palabra.
Una misma frase puede mezclar los dos, y eso es deseable — permite ver si un modelo acierta un
rasgo dialectal y falla el otro.

**Restricción de diseño importante:** solo se marcan palabras que **empiezan** con el fonema
objetivo. En español la `<r>` inicial de palabra es siempre vibrante múltiple, igual que `<ll>` e
`<y>` iniciales son siempre el fonema rehilable. Así el timestamp de palabra de WhisperX marca
directamente el inicio del segmento a medir y se evita montar un alineador forzado (MFA).

Nunca se marca:

- la conjunción `y` aislada — es la vocal [i];
- la `<y>` final de *Jujuy* — es semivocal;
- fonemas intervocálicos (*calle*, *Callao*, *perro*, *mayo*) — audibles, no medibles.

Lo que queda fuera de `targets` no se pierde: va a escucha manual, anotado en `notes`.

La medición se hace sobre el **40–60 % central** del segmento, no sobre los bordes: los
timestamps de WhisperX tienen error de decenas de milisegundos.

#### Largo: escrito vs. hablado

`n_words` es el largo del **texto escrito**, contado como tokens separados por espacio que
contengan al menos un carácter alfanumérico. Se puede declarar en el YAML, donde funciona como
checksum: si no coincide con el conteo derivado, la carga falla.

Las heurísticas de truncamiento y loop (7.5) **no** usan `n_words` sino `n_spoken_words`, que se
deriva de `expected_verbalization` cuando existe. La diferencia no es cosmética: `nor_005` son 8
tokens escritos y 24 palabras habladas, así que un ratio duración/`n_words` marcaría como loop
toda síntesis correcta de esa categoría.

#### `expected_verbalization` — obligatoria con dígitos

La categoría `normalizacion` no tiene medición objetiva sin la forma hablada esperada, así que el
core la exige. Y la exige también para **cualquier** frase cuyo texto contenga un dígito: la
función de normalización no expande números a palabras (`3,5` es "tres coma cinco", no "three
point five"), de modo que sin este campo la referencia del CER contendría dígitos que ningún
hablante dice.

`accepted_alternatives` lista verbalizaciones igualmente válidas — la agrupación de dígitos de un
teléfono admite varias convenciones. El CER se calcula contra la de menor distancia, y de ahí la
disciplina que va con el campo: **tomar el mínimo sobre N formas sólo puede bajar el puntaje**, así
que una alternativa tiene que ser una lectura genuinamente válida y nunca una forma de hacer quedar
mejor a un modelo. La fila registra cuál forma matcheó, que es dato en sí mismo: dice qué convención
usa el modelo y si la sostiene entre seeds.

Los dos campos se escriben **ya normalizados** (minúsculas, sin tildes, sin puntuación) con la
misma función versionada que usa el pipeline de métricas (`rtb.textnorm`), y CI lo verifica: un
valor sin normalizar infla en silencio el CER de toda la categoría.

#### Estabilidad de ids

Los ids son permanentes. Si una frase se corrige, se crea un id nuevo y se deprecia el anterior:
los manifests ya generados apuntan al texto viejo, y reciclar un id mezclaría dos frases
distintas bajo la misma clave. La lista de ids deprecados vive en `rtb.config` y CI rechaza su
reutilización.

**Deprecados:** `sim_001` … `sim_004`. La categoría `s_implosiva` se retiró de v0 porque la
aspiración o elisión de /s/ en coda es un fenómeno de **registro** (habla espontánea), no de
dialecto. SLR61 es lectura cuidada: las referencias no aspiran, el comportamiento correcto del
modelo es no aspirar, y no hay falla que detectar. Recuperarla en v1 solo con referencias de
habla espontánea; ahí sí mide algo interesante — si el modelo copia el registro del clip o
solamente el timbre.

### 5.2 `configs/speakers.yaml`

```yaml
- id: slr61_f_01
  source: slr61
  gender: F
  source_speaker: arf_00295     # código del hablante en el corpus origen
  ref_clips:                    # se pasan al modelo como referencia de voz
    - path: data/refs/slr61/slr61_f_01/ref_00.wav
      duration_s: 11.62
      source_utterances: ["01745620920", "00941298847"]
    - path: data/refs/slr61/slr61_f_01/ref_01.wav      # 2do clip → chequeo de sensibilidad
      duration_s: 10.64
      source_utterances: ["01048417138", "01131884458"]
  control_clips:                # NO se usan como referencia; dan techo, piso y ground truth
    - path: data/refs/slr61/slr61_f_01/ctrl_00.wav
      duration_s: 7.61
      transcript: "..."
      source_utterances: ["01153918696", "02051061628", "01259129668", "01955284393"]
    - {path: "...", duration_s: 5.82, transcript: "...", source_utterances: ["01065483579"]}
  calibration_clips:            # enunciados cortos naturales, para el umbral de 7.5
    - {path: "...", duration_s: 0.93, transcript: "...", source_utterances: ["01708733090"]}
```

**Los clips se construyen, no se editan a mano.** Las utterances de SLR61 son cortas —de 1 a 7 s,
mediana por debajo de 3— así que ninguna sola cubre el presupuesto de referencia que el protocolo
le da a cada modelo. `scripts/build_reference_clips.py` los arma concatenando utterances
**enteras**, en orden, separadas por 250 ms de silencio, sin tocar niveles y sin resamplear: la
normalización de loudness es cosa de la etapa de métricas, sobre una copia.

`source_utterances` es la lista de ids del corpus que componen cada clip. Un solo elemento es una
utterance natural sin tocar; varios, una concatenación. Es lo que hace reproducible el clip para
cualquiera que tenga el corpus, y `--check` verifica que lo que está en disco reconstruya desde
esos ids. Sin ese campo, "usamos 4 hablantes de SLR61" no es una afirmación auditable.

**Cómo se eligen las utterances cuando `source_utterances` viene vacío.** Todos los criterios son
deterministas —sin azar, para que dos corridas coincidan—: la utterance tiene transcripción, no
tiene dígitos y no clipea; `ctrl_00` se arma con las oraciones que **todos** los hablantes elegidos
leyeron, en un orden fijo, que es lo que hace posible el piso de 7.2; el clip de calibración es la
utterance más corta que queda, prefiriendo las de 4 palabras o menos; el resto van de más larga a
más corta, para que cada clip necesite la menor cantidad de empalmes; y ninguna utterance se usa en
dos clips del mismo hablante.

Las tres listas tienen usos distintos y no se mezclan:

- **`ref_clips`** — se pasan al modelo. Cada modelo recorta el clip a su máximo aceptado
  (`max_ref_seconds`), **siempre desde el mismo clip y desde el inicio**; comparar modelos con
  referencias distintas invalida la métrica de similitud. Por eso los clips se construyen por
  encima del `max_ref_seconds` más alto del set de modelos: si un hablante quedara corto,
  entregaría menos segundos de referencia que los otros y dejaría de ser comparable.
- **`control_clips`** — nunca se usan como referencia. Se exigen **dos** como mínimo: el techo de
  similitud es la media entre dos controles reales del mismo hablante, así que con uno no hay
  techo. Su `transcript` es la referencia de las filas de ground truth (5.5), así que tiene que ser
  exacto. El primero de la lista es el clip de contenido controlado (ver 7.2).
- **`calibration_clips`** — enunciados cortos naturales, para el umbral absoluto de duración de
  `enunciados_cortos` (7.5). Van aparte y **no** entran en el techo: el embedding de un clip de
  0.8 s es ruidoso, y promediarlo con los controles largos subestimaría el techo.

Ningún clip comparte audio con otro del mismo hablante. Se valida por ruta y también por id de
utterance, que es lo que importa de verdad: dos rutas distintas pueden contener el mismo audio si
se construyeron de la misma utterance, y un control que contenga audio usado como referencia mide
al modelo contra su propio prompt.

### 5.3 Fila del manifest (`data/manifests/*.jsonl`)

Append-only, una línea por síntesis:

```json
{
  "generation_id": "qwen3_tts__slr61_f_01__yei_003__s1234__ref0",
  "model": "qwen3_tts",
  "model_revision": "a1b2c3d",
  "lib_version": "qwen3-tts==0.4.2",
  "torch_version": "2.5.1",
  "device": "cuda",
  "gpu": "Tesla T4",
  "speaker_id": "slr61_f_01",
  "ref_clip_idx": 0,
  "ref_seconds_used": 3.0,
  "phrase_id": "yei_003",
  "seed": 1234,
  "wav_path": "audio/qwen3_tts/slr61_f_01/yei_003_s1234_ref0.wav",
  "sample_rate": 24000,
  "duration_s": 3.41,
  "ttfb_ms": 412,
  "total_ms": 1230,
  "rtf": 0.36,
  "streaming": true,
  "warmup": false,
  "reference_text": null,
  "error": null,
  "created_at": "2026-08-18T14:22:01Z"
}
```

`warmup: true` marca las síntesis de calentamiento, que no entran en las agregaciones de latencia.
`reference_text` solo se usa en las filas de ground truth (5.5), que no salen de `phrases.yaml` y
necesitan traer su propia transcripción; en las filas de modelo va en `null` y el texto se toma de
`phrases.yaml` por `phrase_id`.

**`model_revision` es obligatorio y debe ser el commit hash de HuggingFace, no el nombre del
repo.** Los pesos cambian sin aviso; sin el hash el benchmark deja de ser reproducible. Para un
modelo local sin repo que citar —Sintética— el equivalente es el **sha256 del contenido** de cada
archivo de pesos: cumple la misma función, que es decir exactamente qué corrió.

Si la síntesis falla, se escribe igual la fila con `wav_path: null` y `error` poblado. Los
fallos son un resultado, no una excepción a esconder.

### 5.4 Fila de métricas (`data/metrics/*.jsonl`)

**El `metric_version` de cada fila nombra exactamente lo que determinó su resultado, y nada más.**
Ni de menos ni de más, y el proyecto se equivocó de las dos maneras antes de fijar la regla:

- **De menos.** `failures` versionaba con `{version de thresholds}/{versión del ASR}`, y la
  taxonomía es sobre todo **código**: el 2026-09-16 dos cambios de regla —la tercera condición del
  loop y la precedencia de `trailing_disagreement`— movieron tasas publicadas sin que ninguna fila
  lo declarara. Ahora lleva `taxonomy<N>` adelante, y `TAXONOMY_VERSION` se sube a mano cuando
  `classify` cambia de conducta. A mano se desactualiza en silencio, así que hay un test que fija el
  sha256 de la fuente de `classify` y `overruns` y falla cuando cambia: ahí se decide si fue
  conducta —subir la versión— o un comentario —repinar el hash—.
- **De más.** `audio_shape` usa tres perillas (`frame_ms`, `silence_threshold_ratio`,
  `silent_peak_ratio`) y versionaba con el `version` global del archivo de umbrales. Mover un
  umbral que nunca lee le invalidaba las 2172 filas: o se releen todos los wav para recalcular lo
  mismo, o el archivo queda con dos versiones describiendo un cómputo idéntico. Pasaron las dos.
  Ahora su versión sale de sus propias perillas: `frame20/silence0.05/peak0.01`.

La regla que queda: **si cambiarlo cambia el número, va en la versión; si no, no.** El
`version` global de `thresholds.yaml` sigue existiendo para las métricas que sí dependen del archivo
entero, y `duration_screen` todavía lo usa — es triage y su JSONL no se commitea, así que la deuda
está acotada y declarada.

Una línea por `generation_id`, con las métricas calculadas. Se une con el manifest por
`generation_id` en la etapa de análisis. Cada métrica se escribe en su propio archivo
(`transcripts.jsonl`, `audio_shape.jsonl`, `failures.jsonl`, `speaker_sim.jsonl`, ...) para poder
recalcular una sin invalidar las otras. Esa separación no es sólo de invalidación: también es lo que
permite **cruzar** métricas sin duplicar datos — el corte por duración de la similitud (7.2) sale de
unir `speaker_sim.jsonl` con `audio_shape.jsonl` por `generation_id`, sin que ninguna de las dos
guarde el dato de la otra.

Dos archivos no son "una fila por síntesis" y por eso llevan un `generation_id` sintético:
`speaker_scale.jsonl`, donde cada fila es un **par de clips reales** (`scale__<tipo>__<etiqueta>`),
y las filas de ground truth de 5.5 (`gt__…`). El esquema es el mismo a propósito, así que ninguna
métrica downstream necesita un caso especial.

### 5.5 Filas de control (audio real)

El pipeline de métricas debe procesar también los `control_clips` y `calibration_clips` reales, con
`model: "__ground_truth__"`. Sin esa fila no hay forma de saber si un WER de 6 % es bueno o malo.
Es un requisito, no un extra.

Las genera `python -m rtb.metrics.ground_truth` desde `speakers.yaml`, en
`data/manifests/ground_truth.jsonl`. Usan el mismo schema y la misma clave de join que una
síntesis, así que toda métrica de abajo las trata como una columna más y nadie escribe un caso
especial.

Tres detalles que las distinguen:

- **El id va prefijado `gt`**, no con el nombre del modelo: `__ground_truth__` empieza con el
  separador y lo contiene, así que no es un token de id válido. El campo `model` lleva el nombre
  completo; el id, la abreviatura.
- **`ref_clip_idx`, `seed` y toda la procedencia van en null.** Ningún modelo corrió y nada se
  condicionó con una referencia. El schema permite esos nulos **sólo** en ground truth, así que una
  fila de síntesis no puede perder su procedencia sin que la validación lo note.
- **El texto viaja en `reference_text`**, porque estos clips no salen de `phrases.yaml` y no hay
  `phrase_id` que consultar. La regla para toda la etapa 2 es: la referencia es `reference_text` si
  está, y si no, la frase del corpus por `phrase_id`.

Los `ref_clips` **no** entran: no tienen transcripción y son el prompt del modelo, así que
transcribirlos mediría el prompt en vez de aportar un punto de referencia. Y la duración se lee del
header del wav, no del YAML: si los dos no coinciden, la corrida aborta, porque una duración
declarada que no es la del archivo también invalida la tasa de habla.

---

## 6. Etapa 1 — Generación

### 6.1 Interfaz del adapter (`generate/base.py`)

```python
class TTSAdapter(ABC):
    name: str
    max_ref_seconds: float
    supports_streaming: bool
    default_sample_rate: int

    @abstractmethod
    def load(self) -> None:
        """Carga el modelo. Todos los imports pesados van ACÁ, no a nivel de módulo."""

    requires_ref_text: bool          # ver 6.4

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def synthesize(
        self, text: str, ref_wav: Path, seed: int, out_path: Path, ref_text: str | None = None
    ) -> SynthesisResult:
        """Sintetiza y escribe el wav. Devuelve timings y metadata, no el audio."""

    def describe(self) -> dict:
        """Versiones de librería, revision del modelo, device. Va al manifest."""
```

`SynthesisResult` es un dataclass simple: `sample_rate`, `duration_s`, `ttfb_ms`, `total_ms`,
`ref_seconds_used`, `ref_text_provided`, `streaming`, `error`. El adapter escribe el archivo; el
core no toca arrays de audio.

`ref_seconds_used` lo reporta el adapter porque es el único que sabe cuántos segundos de
referencia consumió después de recortar a `max_ref_seconds`, y comparar modelos con referencias
distintas invalida la métrica de similitud.

`streaming` es lo que la síntesis **hizo**, no lo que el modelo soporta: de eso depende cómo se lee
su `ttfb_ms`, y un adapter capaz de streamear que corrió entero tiene que decir `false`.

### 6.2 Runner

```
python -m rtb.generate --model qwen3_tts --config configs/models/qwen3_tts.yaml
```

Requisitos:

- **Idempotencia.** Antes de sintetizar, chequea si el `wav_path` existe y si hay una fila en el
  manifest con ese `generation_id`. Si sí, saltea. Trabajando en sesiones cortas, poder cortar y
  retomar sin perder nada es la diferencia entre terminar y no terminar.
- **Warm-up excluido.** Las primeras 3 síntesis de cada sesión no sirven para medir latencia
  (compilación de kernels, cache frío). Se generan **fuera del grid**, con un texto propio y con
  ids `<model>__warmupN`, y quedan marcadas `warmup: true`. Hacerlo así, en vez de descartar los
  timings de las 3 primeras celdas, evita que esas celdas queden sin un timing usable.
- **Seed aplicada a todas las fuentes de aleatoriedad** (python, numpy, torch, cuda). El runner
  registra qué fuentes pudo seedear: "seedeado" con torch ausente no es el mismo experimento.
- Escritura del manifest tras cada síntesis, con flush. No acumular en memoria.
- `--dry-run` que imprima el grid sin generar.
- La grilla se puede subsetear por frase, hablante, categoría, seed o clip de referencia, para
  trabajar en sesiones cortas. Los números publicados salen de la grilla completa.

### 6.3 Latencia — reglas de comparación

Todos los modelos se miden **in-process**, en el mismo hardware, en la misma sesión — Sintética
incluida. Medirla vía HTTP cobraría red y FastAPI, que no son costo del modelo, y la sacaría de la
única tabla donde sí es comparable con los zero-shot.

El adapter de XTTS puede tener un modo `remote` opcional contra un servicio HTTP para demos, pero
**los números publicados salen del modo in-process**.

**Qué entra en el reloj y qué no.** El timing mide texto → audio, con la voz ya condicionada, y
excluye dos cosas:

- **La extracción de latentes de la referencia.** Se calcula una vez por clip y se cachea. Es lo
  que hace un deployment —una voz se condiciona una vez y se reutiliza en cada turno— y además es
  lo único que mantiene las filas comparables: cobrarle el conditioning a la primera de las 135
  síntesis de un hablante convertiría esa fila en un outlier y dejaría a las otras 134 midiendo
  otra cosa. El costo se loguea aparte en vez de esconderse dentro de una fila.
- **La escritura del wav.** El reloj se detiene antes; el I/O de disco no es latencia del modelo.

Si un modelo no permite separar conditioning de síntesis, sus números lo incluyen y hay que
**marcarlo en el README**. Es preferible una nota al pie que una comparación silenciosamente
desparejada.

**TTFB y streaming.** Ojo con confundir streaming de generación con streaming de entrada: el
default de Qwen3-TTS, `non_streaming_mode: false`, sólo simula que el **texto** entra por partes —
el wav completo vuelve igual de una— y los otros modelos reciben la oración entera. El config lo
pone en `true` para que la comparación no cruce dos protocolos de entrada.

Los números publicados salen del modo entero, que es el camino de inferencia
estándar y el que produce el audio que se mide; ahí `ttfb_ms == total_ms`, que no es un relleno:
sin streaming el primer byte de audio existe cuando termina todo. El TTFB real se mide en una tanda
aparte con streaming, sobre un subconjunto, y la fila del manifest queda marcada con `streaming`
para que las dos tandas no se mezclen al agregar latencias.

#### El eje de costo: CPU además de GPU

El RTF se mide en las dos plataformas. En GPU sale del grid completo, que ya se corre para las
métricas de calidad. En CPU sale de un **subset**, porque el costo no necesita el mismo N que la
calidad:

- **8 frases estratificadas por largo × 2 hablantes × 3 seeds.** Las 45 frases se ordenan por
  `n_words`, se parten en 8 estratos de igual tamaño y se sortea una de cada uno, **forzando que el
  primero sea de `enunciados_cortos` y el último de `frase_larga`**: sin eso el sorteo deja afuera
  el turno corto, que es donde el RTF salta (1.035 debajo del segundo contra 0.599 entre 4 y 6 s en
  Chatterbox v3), y medido no cuesta nada — misma banda y sesgo ≈0. La estratificación
  no es cosmética: es lo que hace que la mediana del subset estime la del grid y no otra cosa. Sin
  ella la mezcla de categorías fija el número sin que se vea, porque el RTF depende fuerte de la
  duración.
- **Dos hablantes como mínimo.** El clip de referencia mueve el audio producido ~6 % y el RTF se
  corre por el divisor; con un solo hablante ese sesgo no se promedia, se hereda.
- **El N está derivado, no elegido**, de las 540 celdas por modelo que el grid ya tiene medidas:

  | | variación medida | consecuencia |
  |---|---|---|
  | entre seeds | 0.4 – 1.7 % | promediar no compra nada |
  | entre hablantes | 0.4 – 2.0 % | idem, y dos alcanzan |
  | entre frases | 2.6 – 5.7 % | es la señal, y es lo que fija el N |

  Con 8 frases × 2 hablantes × 3 seeds, el 90 % de los subsets posibles cae dentro de ±0.4 %
  (XTTS-v2), −1.6/+2.4 % (Qwen3-TTS) y −1.7/+1.9 % (Chatterbox v3) de la mediana del grid completo,
  y subir a 15 frases no lo mueve. Lo que sobra de ancho ya no es ruido que se promedie:
  las frases **de verdad** difieren entre sí, y elegir 8 de 45 siempre deja parte de eso afuera. La
  mediana se publica **con esa banda al lado**. Salvedad: la derivación sale del grid de GPU y
  dimensiona una medición de CPU, donde el costo fijo pesa distinto; la propia corrida la puede
  chequear ajustando lo mismo sobre el subset.
- **Se publica la mediana, no una curva.** El RTF contra el largo no se resume en un número estable:
  hay un costo fijo por síntesis más uno por segundo de audio, así que el RTF cae con la duración y
  después se aplana —Chatterbox v3 va de 1.035 debajo del segundo a 0.599 entre 4 y 6 s— y una
  pendiente ajustada sobre un subset no converge ni con 12 frases. Que un turno corto cuesta más que
  uno largo queda **declarado y sin número**, hasta que haya una forma que el subset recupere.
- **Barrido de 1 y 4 threads, fp32 y sin cuantizar.** El RTF en CPU no es un número, y el barrido no
  es precaución: **cuánto compran los hilos es propiedad del modelo, no de la máquina**. Medido,
  XTTS-v2 escala 1.20× entre uno y cuatro threads y Qwen3-TTS y Chatterbox v3 escalan 1.81–1.83×, o
  sea que XTTS casi no paraleliza y es el menos castigado en un equipo de un solo núcleo. Un barrido
  asumido desde un modelo habría dado el número equivocado para los otros dos. Que en producción los
  modelos grandes se desplegarían cuantizados es una salvedad a escribir, no algo que esta tabla
  mida.

**Un modelo que no corre en CPU es un resultado, y tiene tres formas distintas.** La regla se fija
antes de correr, porque después las tres son indistinguibles:

| veredicto | qué pasó | ¿habla del modelo? |
|---|---|---|
| `no_cpu_upstream` | el proyecto no ofrece camino de CPU | **sí** |
| `cpu_con_parche` | corre, con un parche propio | en parte |
| `cpu_no_resuelto` | conflicto de dependencias en nuestro entorno | **no** |

Sólo el primero es una propiedad del modelo. Colapsar los tres en "no corre" le atribuye al modelo
una limitación nuestra.

---

### 6.4 Asimetría de protocolo: la transcripción de referencia

No todos los modelos zero-shot clonan desde lo mismo. XTTS-v2 y Chatterbox clonan **desde audio**;
Qwen3-TTS clona desde un par **(audio, texto)** — su modo ICL, que necesita la transcripción del
clip de referencia. Eso no es un detalle de implementación de un adapter: son dos condiciones
experimentales distintas, y esconderlo haría que la tabla comparara cosas diferentes sin decirlo.

Tres consecuencias, todas obligatorias:

1. **Se le pasa la transcripción.** La alternativa existe —`x_vector_only_mode`, que clona sólo del
   speaker embedding— pero la doc del vendor avisa que la calidad baja, y medir un modelo en una
   configuración degradada que nadie usaría no responde nada. Los `ref_clips` de `speakers.yaml`
   llevan `transcript`, derivado de `source_utterances` y verificado por
   `build_reference_clips.py --check`.
2. **Un modelo que usa el texto NO recorta el audio.** El `transcript` describe el clip completo,
   así que recortar a `max_ref_seconds` deja el texto describiendo más de lo que se oye. Ése es el
   único modo de falla mudo de este modelo: un par desalineado no da error, clona peor.
   Consecuencia medida: Qwen recibe **6.5–11.6 s** de referencia donde XTTS-v2 recibe **6.0 s
   exactos**. Se intentó cortar por prefijo de utterances enteras, que sí mantendría el par
   alineado, y no sirve — da entre 3.70 y 6.55 s según el hablante, y los dos clips de
   `slr61_m_01` son una única utterance de 6.55 s sin punto de corte. Una dispersión de 2.85 s que
   además queda por **debajo** del presupuesto en tres de cuatro hablantes es peor que una por
   encima.
3. **Va en los datos y en el cuerpo del README, no en una nota al pie.** Cada fila lleva
   `ref_text_provided` y `ref_seconds_used`, así que la similitud de hablante (7.2) se puede leer
   controlando por las dos cosas. La sección de metodología del README declara que Qwen3-TTS
   recibe transcripción de referencia y más segundos de audio que los otros modelos.

**El tercer caso: un modelo que recorta de a tres largos a la vez.** Chatterbox Multilingual v3
clona sólo desde audio, pero consume la referencia en tres tramos distintos en la misma llamada
(verificado en `prepare_conditionals`):

| componente | recorte | sobre los clips de 6.5–11.6 s |
|---|---|---|
| tokens de prompt T3 | 6.0 s (`ENC_COND_LEN`/16000) | 6.0 s para todos |
| decoder s3gen | 10.0 s (`DEC_COND_LEN`/24000) | 6.55 a 10.0 s |
| **voice encoder** (speaker embedding) | **ninguno** | **el clip entero** |

El componente sin recorte es justamente el speaker embedding, o sea el más ligado a lo que mide
7.2. Sin intervenir, este modelo vería 11.62 s de una voz y 6.55 de otra, igual que Qwen3-TTS —
pero acá **no hay par (audio, texto) que desalinear**, así que el adapter recorta el wav a
`max_ref_seconds` antes de pasarlo, los tres caminos ven lo mismo y `ref_seconds_used` vuelve a ser
un número con un solo significado. El recorte se escribe una vez por clip, con el presupuesto en el
nombre del archivo para que dos presupuestos no colisionen, y va desde el inicio del clip sin
resamplear ni tocar niveles.

Resultado: **XTTS-v2 y Chatterbox v3 reciben 6.0 s exactos, Qwen3-TTS recibe 6.5–11.6 s.** Dos
contra uno, que es una comparación mucho más fácil de leer que tres conductas distintas — y hace
parcialmente contestable la pregunta A4 de `ANALYSIS.md` sin correr nada extra: si dos modelos que
reciben la misma referencia difieren mucho en similitud, el largo de la referencia explica poco.

`requires_ref_text` en el config del modelo es lo que hace que esto falle temprano:
`load_bench_config` verifica que todos los `ref_clips` que la grilla va a usar tengan
transcripción y aborta en un segundo, en vez de después de bajar un checkpoint de 1.7 B. Qwen
además rechaza un `ref_text` vacío por su cuenta (verificado en su código, que rechaza `None` y
`""`), así que el chequeo del core es fail-fast y no un rescate.

### 6.5 La segunda clase de modelo: la voz que se elige por id

Los tres zero-shot clonan desde un clip. **Sintética no**: es FastPitch multi-hablante + HiFiGAN, y
la voz es un `speaker_id` horneado en los pesos. El clip de referencia de su fila no entra a la
síntesis — es el objetivo del coseno de similitud y nada más (7.2, ANALYSIS A15).

Eso obliga a un parámetro en la ABC. `synthesize` recibe `speaker_id` además de `ref_wav`, el
runner lo pasa siempre —ya lo tiene, va en cada fila del manifest— y los adapters que clonan lo
descartan. La alternativa era deducir el hablante del nombre del directorio del clip, que esconde
un mapeo que **cambia el experimento** adentro de una convención de nombres. El adapter que la usa
declara `selects_voice_by_id`, y su procedencia sale con `voice_selection: speaker_id` para que una
similitud alta no se lea como clonado.

El mapeo hablante del benchmark → voz del modelo vive en el config, no en el adapter, y se valida
al construirlo: pedir una voz que ningún checkpoint declara falla antes de sintetizar, no en la
fila 300. Una fila sin hablante es un error y **no** una voz por defecto: elegir cualquiera daría
540 filas medidas contra el hablante equivocado, y el único síntoma sería una similitud baja que se
lee como un modelo malo.

**Los pesos vienen en dos checkpoints, uno por género, y sólo uno queda residente.** La grilla itera
por hablante, así que la corrida entera cuesta un recambio; tener los dos en memoria duplicaría la
RAM pico que mide 7.3 y el número dejaría de parecerse al de un deployment. El recambio se registra
con su costo y **fuera del reloj**, por la misma razón que el conditioning de XTTS (6.3).

**Tres asimetrías más contra los zero-shot, medidas y declaradas:**

- **La seed no hace nada.** FastPitch es determinístico: tres corridas del mismo texto y hablante
  dieron audio idéntico bit a bit. Las 3 seeds del grid producen tres filas iguales. Se corren
  igual, porque la comparación entre modelos depende de que el denominador sea el mismo, y la
  varianza entre seeds exactamente 0 es un resultado y no un desperdicio — a RTF 0.03 el grid
  entero cuesta minutos.
- **El texto se parte en oraciones y no se puede desactivar**, mientras XTTS-v2 corre con
  `enable_text_splitting: false` a propósito. Sobre las 45 frases el efecto es chico y está medido:
  con el fork **0.1.10** ninguna se parte distinto de como la diría un humano.

  **Con 0.1.9 esto rompía, y el benchmark lo encontró.** Sobre `pro_004` —"Bueno... a ver... ¿cómo
  te explico?"— el segmentador devolvía `['Bueno... a ver..', '.', '¿cómo te explico?']`, y ese
  `'.'` suelto tiraba `IndexError` adentro del modelo: **12 filas de 540 como `hard_error`**, las 4
  voces × 3 seeds. Se reprodujo llamando a la librería cruda igual que la llama el servicio, o sea
  que un pedido con puntos suspensivos a la API se caía del mismo modo. Corregido en 0.1.10 por dos
  lados —el segmentador pega al vecino todo fragmento sin letras, y `forward_tts` dejó de hacer un
  `squeeze(1)` que con un solo token colapsaba el eje equivocado—, y remedido: `pro_004` sale hoy
  con WER 0.000. Se deja escrito porque es lo único que este benchmark le aportó al producto.
- **No hay commit de HuggingFace que citar.** Los pesos son `.pth` locales, así que
  `model_revision` lleva el **sha256 de cada checkpoint más el del vocoder**, que cumple la misma
  función que pide 5.3: decir exactamente qué pesos corrieron.

## 7. Etapa 2 — Métricas

Preprocesamiento común antes de cualquier métrica, en `audio_utils.py`:

- Resample de todo a una frecuencia común. Para similitud de hablante son 16 kHz y no es opcional:
  ver 7.2, donde está medido lo que pasa si se saltea.
- Para métricas espectrales, **limitar banda a 8 kHz**. Sin esto se comparan modelos que salen
  a 24 kHz contra otros a 16 kHz y el resultado es artefacto del sample rate. Cuando el resample de
  destino es 16 kHz, ese límite ya viene por Nyquist y no hace falta filtro aparte.
- Normalización de loudness (`pyloudnorm`, EBU R128, −23 LUFS) sobre una **copia**; los
  originales no se modifican. **No aplica a la similitud de hablante y no se instala para eso**:
  ECAPA normaliza media y varianza de sus propias features, y el coseno del mismo clip escalado
  −12, −6 y +6 dB es exactamente 1.0. Queda pendiente para las métricas espectrales de 7.4, que sí
  la necesitan.

### 7.1 Inteligibilidad

WER **y** CER round-trip con Whisper large-v3 sobre el audio sintético contra el texto fuente.

Reportar los dos: el WER se contamina con la normalización de números y siglas (categoría
`normalizacion`), el CER es más estable. Ambos lados se normalizan con la misma función,
`rtb.textnorm.normalize_text` (minúsculas, sin tildes salvo la ñ, sin puntuación), que vive en el
core justamente para que la validación de los configs y el cálculo de las métricas no puedan
divergir. Está **versionada** (`NORMALIZER_VERSION`, que va a `metric_version`) porque cambiarla
cambia todos los resultados.

La función no expande números a palabras: eso lo aporta `expected_verbalization` en las frases que
tienen dígitos, y el CER de esa categoría se calcula contra esa forma, o contra la
`accepted_alternatives` de menor distancia.

Salvedad a dejar escrita en el README: Whisper mide si se entiende, no si suena rioplatense.

**Transcribir y puntuar son dos pasos, en dos archivos.** Transcribir las 552 filas cuesta cinco
minutos de GPU; calcular WER cuesta milisegundos y se va a repetir cada vez que se toque el
normalizador. Separados, un ajuste del normalizador nunca vuelve a correr Whisper, y el WER se
recalcula en cualquier máquina —CI incluida— porque sólo la transcripción necesita el entorno de
evaluación. El `transcripts.jsonl` guarda el texto **crudo**: guardarlo normalizado ataría las
transcripciones al normalizador y anularía la separación.

**El modelo de ASR es parte de la identidad de la métrica.** La cuantización cambia las
transcripciones, así que un WER sin el modelo y el compute type al lado no es reproducible — el
mismo argumento que `model_revision` en generación. Cada fila los registra y el `metric_version` es
la forma compacta: `large-v3/int8_float16/faster-whisper==1.2.1`.

**Validar el instrumento antes de creerle.** Whisper transcribe los cuatro clips cortos reales
perfecto, incluido uno de 0.75 s y dos palabras. Por eso, cuando escucha "Daily Bermuda" donde la
síntesis debía decir "Dale.", eso es la síntesis siendo ininteligible y no el instrumento quedándose
sin contexto. Sin las filas de ground truth de 5.5 eso no se podía saber.

### 7.2 Similitud de hablante

Coseno entre embeddings de `speechbrain/spkrec-ecapa-voxceleb`: audio sintetizado vs. clip de
referencia usado — **el que la fila declara en `ref_clip_idx`**, no un promedio de los clips del
hablante. El grid varía ese índice para preguntar si la referencia elegida cambia el resultado, y
promediar borraría la respuesta.

El mismo pase con `--unseen` mide contra el **otro** clip del hablante, el que la síntesis nunca
escuchó, y lo escribe como métrica aparte. Es lo que separa identidad de copia del prompt, y el
párrafo del final de esta sección dice cuánto dio.

Corre a 16 kHz **obligatoriamente**. No es higiene: ECAPA se entrenó a esa frecuencia y no
resamplea solo, y pasarle el mismo clip a 22050 Hz da coseno 0.173 contra su propio embedding a
16 kHz — apenas por encima del 0.155 que este benchmark mide entre hablantes distintos. Sin
resample la métrica devuelve ruido, sin error ni warning. Ese resample es además el límite de banda
a 8 kHz de la sección 7, por Nyquist, así que no hay filtro aparte que equivocar.

**Obligatorio reportar escala.** Un 0.72 sin techo ni piso no significa nada, y esta calibración es
lo que separa este benchmark de los que circulan. Medido sobre los cuatro hablantes de v0:

| punto | qué compara | pares | media | rango |
|---|---|---|---|---|
| **techo** | mismo hablante, contenido distinto | 4 | 0.827 | 0.806–0.852 |
| **piso** | hablantes distintos, contenido idéntico | 6 | 0.155 | 0.085–0.221 |
| **techo corto** | mismo hablante, un clip < 1.2 s | 4 | 0.671 | 0.588–0.715 |

- **Techo** — dos `control_clips` reales del mismo hablante, con **contenido distinto** entre sí.
  Distinto a propósito: al modelo se lo mide sintetizando un texto que no es el del prompt, así que
  un techo medido sobre contenido idéntico sería inalcanzable por construcción y haría ver a todos
  los modelos peor de lo que están.
- **Piso** — `control_clips` de hablantes distintos, con **contenido idéntico**. El primer
  `control_clip` de cada hablante es la concatenación de las mismas oraciones, leídas por los
  cuatro, en el mismo orden. Fijando el contenido, la diferencia de coseno que queda es identidad de
  hablante y nada más. El pase **verifica** que ese `ctrl_00` sea de verdad el mismo texto para
  todos y aborta si no: contenido distinto baja el coseno, un piso más bajo ensancha la escala, y
  todos los puntajes normalizados saldrían más altos de lo que son.
- **Techo corto** — mismo hablante, contenido distinto, pero con uno de los dos clips de menos de
  1.2 s (los `calibration_clips`). Se agregó porque **hay un factor de confusión de duración y es
  grande**: sobre las 540 síntesis de XTTS-v2 la correlación entre segundos de habla y coseno es
  0.53, y la mediana sube monótonamente de 0.342 por debajo de un segundo a 0.750 por encima de
  seis. Los `calibration_clips` siguen fuera del techo largo, como decía la versión anterior de esta
  sección, pero ahora tienen su propio punto de comparación en vez de quedar sin escala.

**La comparación entre categorías es inválida a duración despareja.** Dentro de la banda de 2 a 4 s
las ocho categorías que conviven ahí caen todas entre 0.559 y 0.608 — 0.05 de dispersión donde la
tabla por categoría muestra 0.41. O sea que el orden por categoría es, en su mayor parte, largo de
clip, y el reporte lo dice al lado de la tabla. El corte por duración se imprime **antes** del corte
por categoría por eso mismo.

Y no es todo artefacto, que es la conclusión que importa: corregido contra el techo corto —audio
real de la misma duración— el tramo de menos de un segundo igual queda en 0.362 normalizado contra
0.642 del conjunto. Sobre el sesgo de duración hay un déficit real de clonado en enunciados cortos,
que coincide con que sea también la categoría con más fallas de inteligibilidad (35.4 %). Los tramos
de 1 a 4 s no tienen techo pareado, así que su cifra normalizada es una cota inferior.

**Diagnósticos de la construcción de los clips** (no son parte de la escala publicada, y no estaban
en la versión anterior de esta sección). Los `ref_clips` son concatenaciones de utterances enteras
separadas por 250 ms de silencio, y nada había verificado que eso no dañe la identidad del hablante
en el embedding. Dos pares por hablante lo miden:

| diagnóstico | qué compara | media | rango |
|---|---|---|---|
| `ref_vs_control` | referencia concatenada vs. control real | 0.852 | 0.823–0.890 |
| `ref_vs_ref` | las dos referencias concatenadas | 0.880 | 0.853–0.927 |

La referencia concatenada más débil da 0.823, por encima del par real mismo-hablante más débil
(0.806). **La concatenación no daña la identidad**, y el pase falla ruidosamente si algún día deja
de ser cierto. Sin este chequeo, toda la columna de similitud podría estar midiendo nuestra
construcción de clips en vez de al modelo.

Se reporta el **coseno crudo y el normalizado**, y el normalizado dice contra cuál techo. El crudo
depende del checkpoint —que va en el `metric_version` de cada fila, resuelto al commit del snapshot
de HF— así que el normalizado es el único comparable entre benchmarks. El normalizado **no se
recorta a 1**: un modelo por encima del techo se parece a la referencia más de lo que se parecen dos
grabaciones reales del hablante, y eso se ve mejor que se esconde. Medido, pasar el techo resultó
ser el extremo del párrafo que sigue y no un modo aparte (ANALYSIS A5).

**El número lleva una componente de prompt, y va declarada al lado.** Cada síntesis se parece más al
clip que escuchó que a otro clip del mismo hablante: +0.027 en XTTS-v2, +0.032 en Qwen3-TTS y +0.035
en Chatterbox v3 de coseno crudo —0.040 a 0.053 de escala—, con el signo sostenido en 92 a 96 % de
las 540 celdas de cada modelo. No es identidad de hablante: los dos clips de referencia de un
hablante dan 0.853–0.927 entre sí, por encima del techo, así que copiar sólo la voz daría cero de
diferencia. **Se declara y no se resta**: clonar desde ese prompt es la pregunta del benchmark. Como
los tres modelos la tienen casi por igual, el orden entre ellos no depende de eso; el número
absoluto sí (ANALYSIS A5, A10).

**Sintética comparte los cuatro hablantes de SLR61**, así que su fila se lee directo contra las de
los zero-shot: mismos hablantes, misma escala, mismo instrumento. Lo que cambia es qué contesta cada
número — de un lado, qué tan cerca queda un modelo de alguien que escuchó 6 segundos; del otro, de
la persona con la que se entrenó.

Con una salvedad grande: **el audio contra el que se mide Sintética estuvo en su entrenamiento.**
Los clips de `speakers.yaml` no se eligieron con ese criterio, y las utterances held-out de esos
mismos hablantes existen pero hay que buscarlas (ANALYSIS A15).

**Decisión del 2026-09-13: v0 mide Sintética contra los mismos `ref_clips` que los zero-shot, y su
similitud se publica como cota superior.** "Medida contra audio que estuvo en su entrenamiento, la
similitud de Sintética es X" es verdadera y sirve: si X es baja, la conclusión no necesita held-out;
si es alta, no dice nada y queda pendiente. La fila entra en la tabla con marca explícita, no al
pie, porque un número que sólo vive en el texto se lee igual que uno comparable.

**Lo que no se puede escribir es que las dos filas están sesgadas por igual.** El sesgo del
zero-shot está acotado y medido por partida doble: cambiar el clip objetivo mueve la similitud entre
+0.008 y +0.047 en escala, con el efecto por celda debajo del ruido entre seeds, y la componente de
prompt del párrafo anterior es de 0.040 a 0.053 de escala (ANALYSIS A10). Y en buena medida no
es sesgo: clonar desde ese prompt *es* la pregunta. El de Sintética no tiene cota — un fine-tune que
memorizó la utterance puede dar cualquier coseno, y cuánto es exactamente lo que el held-out
contestaría.

Las filas de ground truth no entran en este pase: un `control_clip` no tiene `ref_clip_idx`, y su
similitud contra el resto del audio real del hablante **es** el techo.

### 7.3 Latencia

Agregación de los timings del manifest: mediana y p95 de TTFB y de RTF, por modelo, excluyendo
warm-up y filas con error. Reportar el hardware junto al número, siempre.

**Mediana y p95, nunca la media.** Un outlier de 6 s mueve la media y no dice nada del turno
típico, y la cola de la distribución es justo lo que importa para un agente conversacional.

**La tabla de costo lleva RTF y memoria pico, y ninguna conclusión derivada.** El repositorio
publica los números medidos y la lectura se hace afuera. Declarar un "hardware mínimo" a partir de
dos plataformas sería una extrapolación disfrazada de medición.

| columna | de dónde sale |
|---|---|
| RTF en GPU | grid completo, agregado del manifest |
| RTF en CPU | mediana del subset de 6.3 con su banda, a 1 y a 4 threads |
| RAM pico en CPU | RSS pico del proceso |
| VRAM pico en GPU | pool **reservado** de torch, no el asignado |

**Las dos memorias no comparten columna.** RSS del proceso y pool reservado de CUDA se miden con
herramientas distintas y no significan lo mismo; quien quiere saber si el modelo entra en su equipo
lee la de su plataforma. Las dos son **por corrida** y no por fila: crecen monótonas y las dominan
los pesos del modelo, así que el tamaño del subset casi no las mueve. Que el pool reservado sea la
cantidad a mirar y no el asignado es lo que dejó medido la deriva de Chatterbox.

**Medido el 2026-09-18**, sobre el subset de costo y en la misma placa que produjo los grids —una
RTX 4050 Laptop de 6141 MiB—, con los configs `configs/models/*_gpu.yaml`. Los grids publicados de
los tres zero-shot son anteriores a esta instrumentación y por eso no traen la columna; el de
Sintética sí, y su cifra sale de las 540 celdas:

| | VRAM reservada | VRAM asignada | RSS en CPU, 4 hilos |
|---|---|---|---|
| XTTS-v2 | 2750 MiB | 2057 MiB | 4596 MiB |
| Chatterbox v3 | 3880 MiB | 3545 MiB | 7031 MiB |
| Qwen3-TTS | **5166 MiB** | 4485 MiB | 7093 MiB |
| Sintética | **318 MiB** | 286 MiB | 2271 MiB |

**Qwen3-TTS ocupa el 84 % de una placa de 6 GB** y es el número que decide dónde puede correr;
Sintética entra doce veces más chica que el zero-shot más liviano. El subset alcanza para medirlo:
el pico de Chatterbox con 48 síntesis queda a 0.4 % de lo que su sonda de 90 celdas midió con la
deriva encima (ANALYSIS A14), y XTTS repetido con la placa caliente dio la misma cifra al MiB.

**Una diferencia que cae dentro de las bandas no es un orden.** El subset publica su mediana con la
banda del diseño (6.3), y dos modelos cuyas medianas caen una dentro de la banda de la otra quedan
**empatados y así se reportan**. Romper ese empate con el decimal es inventar precisión: en CPU a 4
threads, Qwen3-TTS da 5.393 y Chatterbox v3 5.578, con p95 de 7.19 y 7.18 — son el mismo número.
Que el orden **cambie entre plataformas** sí es resultado: en GPU Chatterbox va delante de Qwen, en
CPU no se distinguen.

**`LatencySummary.comparable` no ve los threads.** Compara `device` y modo de streaming, y dos
corridas de CPU a uno y a cuatro hilos le resultan comparables sin serlo — el `hardware` de una
corrida de CPU es `cpu:n/a`. Los threads viajan en el `_resources.json` que cada corrida escribe al
lado de su manifest, y la comparación se declara a mano. Es el mismo agujero que la deriva
intra-corrida de ANALYSIS A14: el chequeo automático cubre menos de lo que la comparación necesita.

**La cola no se duplica acá.** Es latencia real para un agente conversacional —un turno que no
cierra— pero vive en la tabla de fallas (7.5), donde ya se publica en segundos, y el eje de costo
remite ahí. El TTFB real todavía no se mide: los números publicados salen del modo entero, donde
`ttfb_ms == total_ms`.

### 7.4 Acento rioplatense — v1

Dos fenómenos, la misma maquinaria: sobre el segmento que abre cada `target` (posición inicial de
palabra, timestamp de WhisperX), midiendo el 40–60 % central.

**Yeísmo** — features:

1. Centro de gravedad espectral del segmento.
2. Ratio de energía 2.5–8 kHz sobre energía total.
3. Presencia de sonoridad (HNR o autocorrelación).

Con eso se clasifica cada realización en:

- **[ʝ] / [j]** — aproximante palatal, poca energía alta → no rioplatense
- **[ʒ]** — fricativa sonora → rioplatense tradicional
- **[ʃ]** — fricativa sorda → porteño contemporáneo

La métrica por modelo es el porcentaje de realizaciones que caen en [ʒ] o [ʃ].

**Rótica** — las mismas features más una específica:

1. **Número de oclusiones** — picos y valles periódicos en la envolvente de amplitud,
   típicamente 20–35 Hz de modulación. Es el feature propio de la categoría.
2. Ratio de energía en alta frecuencia — detecta asibilación (fricción).
3. Duración del segmento.
4. Sonoridad (HNR).

**Cómo se reporta la rótica — importante.** El resultado se expresa como **distancia respecto de
la distribución de los hablantes reales de SLR61**, nunca como "más fuerte" o "más débil" que
otras variedades. Motivo: la afirmación de que el rioplatense trina con más fuerza que el
mexicano o el colombiano es discutible en la literatura. Los marcadores claros son la asibilación
(vibrante con fricción) y el grupo /tɾ/ cuasi-retroflejo, pero la asibilación es más
característica del noroeste argentino que del porteño. Anclando la métrica a la distribución
empírica de SLR61 —cuyos hablantes son porteños— el resultado es válido sin depender de que la
premisa dialectal sea correcta.

En ambos casos los umbrales **se calibran sobre los `control_clips` reales de SLR61**, que dan la
distribución de referencia, y se validan a oído sobre una muestra aleatoria. Guardar los umbrales
calibrados en un YAML versionado, no hardcodeados.

### 7.5 Tasa de fallo

Contar y reportar como porcentaje del total:

- **Alucinación** — CER por encima de un umbral alto, o transcripción con contenido ausente
  del texto fuente. Palabras de más **en el texto**: no incluye la cola de audio, que es otra falla
  y se cuenta aparte.
- **Cola** (`trailing_audio`) — sonido audible después de la última palabra transcripta. Se reporta
  como segundos, no sólo como booleano.
- **Truncamiento** — `duration_s` muy por debajo de lo esperado según `n_spoken_words`. Si además
  hay sonido después de la última palabra, la fila no es esto: es desacuerdo de cierre.
- **Loop** — n-gramas repetidos en la transcripción, o duración muy por encima de lo esperado.
- **Error duro** — filas con `error != null`.
- **Desacuerdo de frontera** (`boundary_disagreement`) — el WER ve inserciones y el CER las
  desmiente. **No es una falla**: es un veredicto propio que saca la fila del ranking sin
  absolverla.
- **Desacuerdo de cierre** (`trailing_disagreement`) — faltan palabras al final **y** hay sonido
  justo ahí. Tampoco es una falla, por la misma razón: truncamiento y cola no pueden ser verdad a
  la vez, así que ninguno de los dos es la respuesta.

Casi ningún benchmark reporta esto y es lo que más importa para producción. Debe ser una
columna de primer nivel en la tabla, no una nota al pie.

**Casi nada de la clasificación es un umbral.** La distancia de edición se descompone, y la
descomposición *es* la taxonomía: palabras de más son alucinación, palabras faltantes al final son
truncamiento, repetición **con** material extra es loop, sonido después de la última palabra
transcripta es **cola** (`trailing_audio`), y un archivo sin habla es silencio. Sólo quedan dos números
elegidos —cuántas palabras faltantes al final cuentan como cortarse, y cuánto sonido posterior
cuenta como cola—, y los dos se barren con `--sensitivity`.

La regla del loop exige **tres** condiciones, y las tres salieron de falsos positivos reales de la
misma frase. Las dos primeras: `nor_005` volvió como "…punto com punto com", que repite un bigrama
y **no agrega nada** — el modelo dijo mal el dominio, que es una sustitución.

La tercera, agregada el 2026-09-16, es la que la frase enunciaba desde el principio: **un
decodificador en loop repite _y se pasa de largo_**. Con sólo "repite y agrega una palabra",
`nor_005` leyendo "uno uno" por "once" marcaba loop en 12 filas — una palabra de más sobre 21, con
la frase entera y correcta alrededor. Se mide como palabras de la hipótesis sobre palabras de la
referencia, y **la meseta es ancha**: entre 1.1 y 1.5 las filas marcadas no se mueven. Los loops
reales están en 2.0 —un enunciado corto dicho dos veces, "Dale." → "Dale, dale."— y en 3.11, que es
XTTS yéndose por las ramas 28 palabras; el falso positivo estaba en 1.05. El corte va en 1.3, el
medio de la meseta, y el barrido sale con `--sensitivity`.

Al agregarla quedan 8 filas marcadas contra 20: los tres modelos zero-shot no se mueven un decimal
—sus loops son todos desbordes de verdad— y Sintética pasa de 2.4 % a 0.6 %. Que la condición sólo
toque a un modelo es lo que dice que no es un umbral elegido para que dé lindo.

**Nota sobre el alcance de la detección.** `max_ngram_repeats` mira n-gramas de hasta 4 palabras, así
que los loops que la regla agarra son de enunciados cortos repetidos. Un párrafo repetido entero no
lo ve; hasta ahora no apareció ninguno.

**`boundary_disagreement` precede a `hallucination` en la cadena de precedencia**, porque son
exactamente las filas que sin él caen ahí. Cuando el WER muestra inserciones y el CER se queda muy
abajo no hay material extra: hay desacuerdo sobre dónde termina una palabra. Pegando el par sin
espacios, `llamaloyaquelallave…` contra `yamaloyaquelallave…` son 2 caracteres de 32 —CER 0.05
contra WER 0.22— y la divergencia entre las dos medidas *es* la señal. Las dos ya se calculan.

El veredicto **no absuelve al modelo ni lo condena**: manda la fila a escucha. Puede ser un
rehilamiento tan marcado que el ASR eligió otra grafía, o una aproximante que el ASR escribió como
`y`, y la regla no los distingue ni debe pretenderlo — eso es la métrica de acento de 7.4. Llamarlas
alucinación está mal en los dos casos. Sin este veredicto esas filas se irían a limpio, y el único
lugar donde el instrumento roza el fenómeno dialectal quedaría invisible.

**`trailing_disagreement` precede a `truncation` y a `trailing_audio`**, que son las dos reglas
que sin él se reparten esas filas. Faltan palabras al final y hay sonido donde deberían estar: o el
ASR no las oyó o el modelo las dijo mal, y desde el texto se ven igual. El piso es **1 palabra
faltante** y no `trailing_deletions_min`, porque la palabra final suelta es justamente el caso que
ese umbral evita llamar truncamiento — y con sonido al lado deja de ser un descarte para pasar a ser
la pregunta.

Las dos filas que marca sobre v0 son las dos respuestas posibles, y por eso el veredicto manda a
escuchar en vez de decidir: en `qwen3_tts__slr61_m_02__lex_004__s1234` el modelo dice "che" y
Whisper no lo transcribe —falla del instrumento—, y en
`chatterbox_v3__slr61_f_02__nor_002__s5678` el modelo dice "tres com por cinco" por "3,5 %" y
Whisper transcribe lo que puede —falla del modelo, pero de pronunciación y no de truncamiento—.

El piso de todo esto es el audio real: sobre los 12 clips de ground truth, este mismo pipeline da
**WER y CER exactamente 0.000 y cero inserciones**. Por eso lo que se mide arriba de cero es
atribuible a la síntesis, y por eso la fila de audio real va en la misma tabla y no al pie.

**Sin audio esto es un screening, no la tasa publicada.** `rtb.metrics.duration_screen` corre sobre
el manifest en dos segundos y sirve para saber si una corrida salió sana antes de gastar una hora en
el pase de ASR. No es un resultado, y **su JSONL no se commitea**, por dos razones que no se
arreglan afinando umbrales:

1. `duration_s` es el largo del wav completo, así que incluye el silencio que el modelo emite antes
   y después del habla. Los clips de referencia vienen recortados, así que su duración es habla
   pura. La vara y lo medido no son la misma cantidad, y de ahí que los ratios sintéticos queden
   sistemáticamente arriba de 1.0 (1.10 a 1.41 por categoría en XTTS-v2) sin que ningún modelo
   hable lento. Medido: un clip real tiene 0.01 s de silencio final, el sintético 0.23 s. **Esta
   tiene arreglo y está más abajo**, al precio de necesitar el ASR.
2. Pasarse de largo es **a la vez** loop y alucinación. Distinguirlos necesita la transcripción:
   n-gramas repetidos es loop, contenido nuevo inventado es alucinación.

Las dos se resuelven en la etapa que carga audio: con la envolvente de energía se separa habla de
relleno —y el relleno pasa a ser su propia métrica, porque un modelo que emite 2 s de cola es peor
para un agente aunque el habla sea perfecta— y con la transcripción se separa loop de alucinación.

#### La anatomía del audio tiene cuatro tramos, no dos

`audio_shape` separa habla de relleno y es ciego al transcript **a propósito**: sabe dónde hay
sonido, no qué dice ese sonido. Esa ceguera cuesta una distinción, y es la cara. El audio que sigue
**después** de la última palabra transcripta está por encima del umbral de silencio, así que cuenta
como habla: un modelo que contesta "¿Cómo?" y después hace ruido dos segundos queda registrado con
dos segundos de habla y un relleno impecable.

El corte en `last_word_end_s` da la cuenta que el reporte necesita:

```
total = silencio inicial + habla pedida + cola + silencio final
```

Es una cantidad **derivada al reportar** (`metrics/anatomy.py`), uniendo `audio_shape.jsonl` con
`transcripts.jsonl`, y **no** un campo nuevo en `audio_shape`: ese pase corre en el entorno de audio
y no puede empezar a depender de que el ASR haya corrido (sección 3).

Medido sobre los tres modelos de v0, mueve conclusiones y no decimales:

| | XTTS-v2 | Qwen3-TTS | Chatterbox v3 |
|---|---|---|---|
| habla según `audio_shape`, p50 | 2.69 s | 2.24 s | 2.80 s |
| **habla pedida**, p50 | 2.48 s | 2.08 s | **2.07 s** |
| cola, p50 | 0.18 s | 0.16 s | **0.30 s** |
| cola, p95 | 1.10 s | 0.34 s | **2.40 s** |
| filas con más de 0.5 s de cola | 9.4 % | 0.7 % | **43.3 %** |
| no-habla, p50 | 0.38 s | 0.60 s | **1.48 s** |

Chatterbox parecía hablar más largo que Qwen —2.80 contra 2.24— y descontando la cola hablan igual,
2.07 contra 2.08.

**La cola se publica en segundos, junto al relleno y al RTF.** Un booleano en la tabla de fallas
pierde el tamaño de un defecto que se mide en segundos, y para un agente conversacional esos
segundos son latencia que el RTF no captura: un turno que no cierra. La mediana sola tampoco
alcanza, porque lo grave está en la cola de la distribución.

**El screening de duración juzga el habla pedida, y eso le cuesta la propiedad de correr sin ASR.**
Con la anatomía disponible la vara y lo medido son por fin la misma cantidad, y las dos razones del
sesgo de arriba dejan de aplicar a la primera. Lo que cambió: `too_long` pasa de 7.0 % a 3.5 % en
XTTS-v2, de 1.7 % a **0.0 %** en Qwen3-TTS y de 9.6 % a 1.7 % en Chatterbox v3. Qwen es el caso que
aclara qué medía el número viejo: **nunca produce habla del largo equivocado**, y todas sus marcas
eran cola — el mismo defecto que la taxonomía ya contaba con otro nombre. Juzgar el habla además
destapa `too_short` por primera vez (0.6 %, 1.9 %, 0.9 %): el relleno venía sosteniendo por encima
del piso a clips cortos. Sin transcripts el screening sigue corriendo sobre el archivo, que es para
lo que se construyó —triage antes del pase caro—, y cada fila declara qué cantidad juzgó.

**El nombre del veredicto es `trailing_audio` y el cambio no es cosmético.** "Cola alucinada" mezcla
dos fallas distintas en una etiqueta: palabras de más en el texto y audio que sigue después de la
última palabra. Son independientes —el 82.2 % de las filas marcadas de Chatterbox v3 tiene WER
exactamente 0.000, o sea texto perfecto *y* sonido de sobra— y leer la etiqueta mezclada hacía que
un defecto real pareciera un falso positivo. Qué **es** esa cola (habla inventada, ruido, o las dos)
no sale del texto y va a escucha humana; que es audible sí está medido: el 100 % de las 268 filas
marcadas supera los −30 dB del pico de su propio clip, con mediana ≈ −12 dB.

**El umbral de la cola se barre como los de duración**, y el barrido dice algo que la tasa sola no:

| `tail_after_last_word_s` | XTTS-v2 | Qwen3-TTS | Chatterbox v3 |
|---|---|---|---|
| 0.2 s | 35 % | 24 % | 60 % |
| 0.3 s | 15 % | 5 % | 51 % |
| **0.5 s (en uso)** | **8 %** | **0 %** | **46 %** |
| 1.0 s | 5 % | 0 % | 38 % |
| 2.0 s | 1 % | 0 % | 12 % |

La tasa absoluta depende del umbral y no se publica sola; el **orden** sobrevive a todo el rango, y
eso sí es un resultado.

**Medido, ya no hipótesis** (`audio_shape.jsonl`, 552 clips): el audio real tiene 0.038 s de relleno
en la mediana; el sintético de XTTS-v2, 0.208 s, y hasta 2.92 s. Toda la diferencia está al final
—0.028 s contra 0.192 s— y no en el ataque. Corrigiendo el ratio por eso, la mediana global baja de
1.25 a 1.14: el relleno explicaba **menos de la mitad** del exceso. Lo que queda es tempo real, y
también es un resultado: XTTS habla más lento que los hablantes de referencia, sobre todo en
`toponimia` (1.32) y `frase_larga` (1.22), donde el relleno es despreciable frente al largo. En
`enunciados_cortos` el ratio corregido sigue en 2.28, o sea que ahí el problema nunca fue relleno.

Consecuencia práctica: **el umbral absoluto de enunciados cortos ya se puede calibrar a oído.** La
cantidad que se umbraliza dejó de moverse —es el habla pedida, no la duración con relleno— así que
un número ajustado ahora no hay que reajustarlo después.

**La duración esperada sale de los hablantes reales, no de una constante.** La tasa se deriva de
los `control_clips` y `calibration_clips`, descontando los 250 ms de silencio que la concatenación
insertó, y es **por hablante** —los cuatro van de 0.256 a 0.291 s/palabra, y cobrarle a uno lento
la tasa de uno rápido se lee como loop—. Dato que hay que tener presente al leer los ratios: los
clips de SLR61 vienen con silencios recortados y el audio sintetizado trae ataque y cola, así que
los ratios sintéticos quedan sistemáticamente **arriba de 1.0** (1.10 a 1.41 por categoría en
XTTS-v2). Eso es la referencia sin relleno, no los modelos alargando.

**Los umbrales van en `configs/thresholds.yaml`, versionados**, y su `version` viaja en el
`metric_version` de cada fila: mover un umbral cambia la tasa publicada y eso tiene que ser
detectable. Y hay que **reportar la sensibilidad** junto con la tasa (`--sensitivity`): un número
que se mueve del 85 % al 15 % barriendo umbrales plausibles no está listo para publicarse, y la
única forma de saberlo es barrer. Reportar el barrido al lado de la cifra es lo que evita que un
umbral elegido se lea como un hecho medido. Sobre v0 el criterio ya separó los umbrales en dos: el
de ratio y el de frontera están sobre una meseta —duplicar el corte mueve unas pocas filas— y el
del margen absoluto y el de la cola no, y esos dos se publican con su barrido al lado.

**Excepción — `enunciados_cortos`.** Con 1–3 palabras el ratio duración/`n_spoken_words` no
discrimina: el término constante (ataque, cola de silencio) domina el total. Para esa categoría
se usa un **umbral absoluto de duración**, más detección de energía después del final del contenido
transcripto — que es la forma de cazar la cola de audio, el modo de falla típico de los modelos
autorregresivos con inputs muy cortos.

El umbral se calibra contra los `calibration_clips`, no contra los `control_clips`: de un clip de
6 s no se deriva cuánto debería durar una palabra. Salvedad a dejar escrita: los enunciados más
cortos que tiene SLR61 son de 2 a 4 palabras y los de esta categoría son de 1 a 2, así que el
umbral queda aproximado y hay que validarlo a oído antes de reportar la tasa.

### 7.6 Niveles de medición — declarar en el README

No todas las categorías se miden igual, y publicar sin decirlo es lo que hace que un benchmark
no se pueda auditar. La tabla va en el README:

| Nivel | Categorías | Método |
|---|---|---|
| **Automático** | yeísmo, rótica, enunciados cortos, frase larga, léxico, toponimia, anglicismos | WER/CER contra el transcript; taxonomía de fallas por descomposición del alineamiento |
| **Semi-automático** (v1) | voseo, prosodia | Duración relativa de núcleos vocálicos; pendiente de F0 final |
| **Solo escucha** | **normalización** (ver abajo), Callao (`top_001`), par mínimo (`rot_005`), /tɾ/ (`rot_004`), naturalidad | Etiquetado ciego binario |

**`normalizacion` pasó a "solo escucha", y con evidencia.** Whisper devuelve los números a dígitos:
una síntesis que dice correctamente "quince mil setecientos cincuenta" vuelve como `15 750`, y
contra una referencia en palabras eso son 5 errores de 11 sin que el modelo se haya equivocado. Las
60 filas de la categoría quedan marcadas `unscorable` — detectado, no hardcodeado: cualquier fila
donde un lado tiene dígitos y el otro no, o donde el ASR escribió un símbolo que se pronuncia
(`%`, `$`), tiene un alineamiento que mide la convención del ASR y no al modelo.

Probamos la salida obvia y no sirve: con `initial_prompt` Whisper sí escribe los números en
palabras, pero no arregla las fechas (`25.05.2026` sigue igual) y **degrada otras cosas** — "AFIP"
pasó a "Zafip". Cambia un artefacto por otro, así que el instrumento queda uniforme y la categoría
se declara no medible por ASR en v0.

Lo que **sí** se arregló, porque ahí el número es incidental y no el objeto de medición:
`expand_small_cardinals` canonicaliza dígitos sueltos de 0 a 100 al puntuar, así que "a las 3"
contra "a las tres" se mide normal. Sin eso se descartaba el 25 % de `anglicismos` por una frase
que dice "a las tres", que no tiene nada que ver con lo que esa categoría mide.

En v0, voseo y prosodia van a **escucha humana**. Con 1 seed son 9 frases × 4 hablantes ×
3 modelos = 108 clips, unos 20 minutos de etiquetado. Condiciones del protocolo, que son lo que
hace que el número signifique algo:

- orden aleatorio y modelo oculto;
- etiqueta **binaria**, no escala 1–5 — una escala no es reproducible entre personas;
- `control_clips` reales mezclados sin avisar. Si el audio real se etiqueta mal, el criterio se
  corrió y la tanda se descarta.

**Lo que v0 no afirma.** Con 7.4 en v1, ninguna categoría mide la **realización** dialectal. El WER
de `yeismo_rehilado` es casi ciego al yeísmo: la ortografía española es yeísta y el 97.2 % de los
onsets vuelve como `ll…`, que cubre [ʃ], [ʒ] y [ʝ] por igual, así que un modelo puede hacer un [ʝ]
castellano en toda la frase y sacar WER 0.000. Los números de la categoría dicen que se entendió, no
que suene rioplatense, y el README lo dice con esas palabras. Lo publicable en v0 es identidad de
hablante, inteligibilidad, tasa de fallo y costo.

---

## 8. Entornos y dependencias

`pyproject.toml` declara **solo** el core (pydantic, pyyaml). Los entornos de trabajo se
levantan aparte:

- **Colab** (camino principal): un notebook por modelo. El aislamiento es gratis — cada runtime
  instala lo suyo y muere. Los notebooks montan Drive o escriben a un bucket para persistir
  `audio/` y `data/manifests/`.
- **Local**: scripts en `scripts/` con dependencias inline PEP 723, ejecutados con
  `uv run scripts/gen_qwen3.py`. Entorno efímero y aislado por script, sin conda ni Docker.

Los entornos de métricas van sin `.lock.txt` por ahora, y el motivo es distinto en cada uno: el de
ASR porque su identidad de instrumento viaja en cada fila (`metric_version` con modelo, cuantización
y versión de librería), y el de similitud porque es CPU-only y por lo tanto se instala igual en
cualquier máquina — no hay variante de CUDA que congelar. Los dos declaran adentro las versiones con
las que se verificaron y la fecha.

Dos archivos por modelo, con roles distintos, nombrados como el modelo:

- **`envs/<model>.txt` — la receta.** Los pines mínimos, cada uno con el motivo escrito. Es lo que
  hay que leer para instalar en otra máquina, porque parte de la resolución no es automática: la
  variante de CUDA de `torch` depende del driver, y el wheel de PyPI apunta al CUDA más nuevo sin
  chequearlo. Si el driver es más viejo, `torch.cuda.is_available()` da `False` en silencio y la
  corrida se va a CPU, donde la latencia no significa nada.
- **`envs/<model>.lock.txt` — el registro.** `uv pip freeze` de la máquina que generó el manifest,
  con el índice de PyTorch declarado adentro (un pin `+cu126` no existe en PyPI). Documenta qué
  corrió, no qué instalar: copiarlo a ciegas en otra placa puede dejar la GPU sin usar.

Los dos se validan contra los resultados. `tests/test_manifest_provenance.py` verifica que todas
las filas de un modelo compartan un solo stack y un solo checkpoint, que el `model_revision` sea un
hash de commit y no un nombre de repo, y que el `lib_version` y el `torch_version` de las filas
coincidan con lo que pinnea el lock. Sin eso, "estos números salieron de este stack" es una
afirmación que nadie chequea y que se podre sola en cuanto alguien actualice un paquete y regenere
medio grid.

`envs/README.md` explica cómo levantar cada entorno.

**No agregar Docker en v0/v1.** No sirve en Colab y no compra nada todavía.

---

## 9. Qué se versiona

| Artefacto | Destino | Motivo |
|---|---|---|
| Manifests y métricas (JSONL) | Git | Son el resultado. Pocos MB. Auditables. |
| Screening de duración (JSONL) | Ninguno | No es un resultado, y se recalcula del manifest en dos segundos. |
| ~30 wavs curados (2-3 por categoría) | Git | Para que el README suene sin bajar nada. |
| 1620 wavs completos (~400 MB) | HuggingFace Dataset | Gratis, versionado, y es contenido de portfolio en sí mismo. |
| Modelos, checkpoints, caches | Ninguno | `.gitignore`. |

Nada de Git LFS: la cuota gratuita de GitHub se agota enseguida y complica los clones.

---

## 10. CI

GitHub Action, sin GPU, que en cada push:

1. Corre los tests del core.
2. Valida el schema de todos los JSONL en `data/`.
3. Valida que `configs/phrases.yaml` cumpla las restricciones: ids únicos y no deprecados,
   categorías conocidas, `targets` presentes en el texto y en posición inicial de palabra con el
   fenómeno correcto, `expected_verbalization` presente donde hace falta y ya normalizada, y
   `n_words` coincidente con el texto.
4. Verifica que `rtb.schema` y `rtb.manifest` sean importables sin torch instalado.

El punto 4 es el que protege el principio de la sección 3 de erosionarse con el tiempo.

---

## 11. Orden de implementación sugerido

1. `schema.py`, `manifest.py`, `config.py` + tests. Sin esto bien resuelto, los tres adapters se
   reescriben después.
2. `generate/base.py` con la ABC y un `DummyAdapter` que genere ruido blanco, para poder probar
   el runner completo sin GPU.
3. `generate/runner.py` con idempotencia y `--dry-run`, validado contra el dummy.
4. Adapter de XTTS (entorno ya conocido) → primera corrida real end-to-end.
   El adapter y sus tests se escriben en CPU; la corrida real necesita la máquina con GPU.
5. Latencia (resultado publicable) y el screening de duración (triage, no resultado). La tasa de
   fallo **no** es "solo el manifest": alucinación y loop necesitan la transcripción, y la duración
   sin audio confunde habla con relleno, así que la categoría entera depende del paso 6.
6. Métricas de inteligibilidad y similitud de hablante, incluyendo las filas de control. Cierra la
   tasa de fallo: separa loop de alucinación, y separa habla de relleno para que el umbral de
   duración se pueda calibrar a oído una sola vez, sobre la cantidad correcta.
   Hacer la similitud **antes** de agregar modelos resultó ser la decisión correcta y por una razón
   que no era la esperada: era el único paso que podía invalidar el protocolo —los clips de
   referencia concatenados (7.2) y la escala techo/piso— y de paso destapó el factor de confusión de
   duración, que habría contaminado la lectura por categoría de los tres modelos.
7. Adapters de Qwen3-TTS y Chatterbox. Qwen necesita además la **transcripción** del clip de
   referencia para clonar zero-shot: hay que poblar `transcript` en los `ref_clips` (se regenera
   exacto desde `source_utterances`), pasar el texto por la ABC, validar que no llegue vacío —un
   Qwen con transcripción vacía clona mal en silencio— y declarar en el README la asimetría de
   protocolo, porque un modelo va a recibir audio + texto y el otro sólo audio.
8. **El eje de costo y la fila de Sintética.** Tres piezas independientes, y ninguna toca el
   pipeline de calidad: el subset de RTF en CPU para los tres zero-shot (6.3), la instrumentación de
   memoria pico por corrida (7.3), y el adapter in-process de Sintética, cuya similitud se mide
   contra los mismos `ref_clips` y se publica como cota superior (7.2). Conviene arrancar por lo
   más barato, que además dimensiona el resto: derivar el N del subset de la varianza que el grid
   ya tiene medida. Es core y no necesita GPU.
9. Reporte y README con resultados de v0. Las dudas abiertas sobre esos resultados no frenan el
   README: van a `ANALYSIS.md`, con su costo y con qué significaría cada resultado posible, y el
   README declara las que todavía no se cerraron. Un número publicado con su salvedad medida vale
   más que un número que espera seis meses a estar perfecto.
10. → v1: métrica de acento, modelos adicionales, dataset en HF.

---

## 12. Criterios de calidad del código

- Type hints en todo el core. `mypy` en CI si no cuesta mucho.
- Los adapters no deben tener lógica de negocio: solo traducen entre la ABC y la API del modelo.
  Toda decisión de protocolo experimental vive en el runner o en configs.
- Errores de un modelo no abortan la corrida: se registran y se sigue.
- Logging estructurado, no `print`.
- Ninguna ruta hardcodeada. Todo sale de configs o de CLI.