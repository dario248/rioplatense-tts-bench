# ANALYSIS.md — análisis pendientes sobre datos ya generados

Registro de lo que los resultados de [`SPEC.md`](SPEC.md) dejaron sin resolver.

---

## Dónde estamos

> **Punto de entrada al abrir sesión.** Una línea por campo y nada más: si crece con hallazgos y
> números deja de servir para lo único que tiene que hacer. El detalle vive en las entradas. Se
> reescribe al cerrar cada sesión.

- **Estado:** **v0 cerrado y publicado** el 2026-10-03. Los pasos 1 a 9 del SPEC están hechos y el
  10 es v1.
- **Último hito (2026-10-17):** infraestructura, y no mueve ningún número. Se congelaron los locks
  de Qwen, Chatterbox y Sintética —antes lo tenía sólo XTTS— y entró el CI que pedía SPEC 10,
  tests en 3.10, 3.11 y 3.12 más mypy, con lo que la procedencia de los cuatro manifests se
  chequea en cada push. → SPEC 8 · 10.
- **Próximo paso: medir la precisión de los timestamps de palabra.** Es lo que decide el tamaño de
  7.4, el diferencial de v1: el SPEC pide el 40–60 % central del onset y lo que hay son timestamps
  de palabra de faster-whisper. Si no alcanzan, hace falta alineamiento forzado. En paralelo va
  A18, que son minutos. → SPEC 7.4 · A18.
- **Bloqueado para publicar:** nada.
- **Lo que v0 no afirma:** nada sobre la realización dialectal, y son **dos** las categorías ciegas a
  su propio fenómeno. Lo publicable es identidad, inteligibilidad, fallas y costo. → A1.
- **Abiertos:** ★★ A6 · A18 · ★ A7 · A8 · A9 · A17 · A11 **diferida**. Ninguno frena v1.
  → §entradas.
- **Pendiente de medir:** nada de v0. Lo que queda para v1 está en SPEC 2, y cada entrada abierta
  trae su costo escrito.

---

Cada entrada dice qué se pregunta, **por qué importa**, qué datos necesita, cuánto cuesta y **qué
significaría cada resultado posible** — eso último es lo que hace que el análisis se pueda retomar
meses después sin volver a razonar el contexto, y lo que evita el peor final: correr algo, ver un
número y no saber si es bueno.

> **Nota de idioma:** este documento va en español, igual que `SPEC.md`. El `README.md` público, los
> docstrings y los identificadores de código van en inglés.

**Qué es y qué no es este archivo:**

- **No es** `SPEC.md`. Ahí va el diseño: qué se construye y por qué. Acá van preguntas sobre lo ya
  construido. Cuando un análisis cambia el diseño, el cambio se escribe en el SPEC y la entrada de
  acá se cierra apuntando a la sección.
- **No es** el `README.md`. Ahí van los resultados publicables. Acá van las dudas sobre ellos,
  incluidas las que todavía no sabemos resolver.
- **No es** una lista de tareas de implementación. Esas viven en el orden del SPEC sección 11. Acá
  hay mediciones, no features.

**Regla de oro del archivo:** una entrada se cierra con un **número**, no con una opinión. Si se
cierra sin poder medir, se dice explícitamente que quedó sin medir y por qué.

**Prioridad:** ★★★ puede invalidar un número publicado · ★★ cambia una conclusión · ★ agrega
información.

---

## Estado de los datos en disco

Al 2026-09-18, contra `configs/grid.yaml` (45 frases × 4 hablantes × 3 seeds × 1 clip de
referencia). **Los cuatro modelos de v0 están generados y medidos.**

| archivo | líneas | ids | contenido |
|---|---|---|---|
| `data/manifests/xtts_v2.jsonl` | 544 | 543 | 540 del grid + 3 warm-up + 1 reintento (ver abajo) |
| `data/manifests/qwen3_tts.jsonl` | 543 | 543 | 540 del grid + 3 warm-up |
| `data/manifests/chatterbox_v3.jsonl` | 543 | 543 | ídem |
| `data/manifests/sintetica.jsonl` | 1086 | 546 | el grid con el fork 0.1.9 y otra vez con el 0.1.10 |
| `data/manifests/*_ref1.jsonl` | 543–600 | ídem | el grid con el segundo clip de referencia (A10). Chatterbox trae 60 warm-up, uno por reinicio (A14) |
| `data/manifests/ground_truth.jsonl` | 12 | 12 | audio real (8 control + 4 calibración) |
| `data/metrics/transcripts.jsonl` | 2172 | | 540 × 4 + 12 |
| `data/metrics/audio_shape.jsonl` | 2172 | | envolvente de energía |
| `data/metrics/speaker_sim.jsonl` | 3780 | | coseno contra el clip que la fila escuchó: 540 × 7 |
| `data/metrics/speaker_sim_unseen.jsonl` | 3780 | | el mismo audio contra el **otro** clip del hablante (A5, A10) |
| `data/metrics/speaker_scale.jsonl` | 22 | | techo, piso, techo corto, diagnósticos |
| `data/metrics/failures.jsonl` | 2172 | | WER/CER + taxonomía |
| `data/latency/*_cpu_t{1,4}.jsonl` | 51 | | el subset de costo en CPU, a 1 y a 4 hilos (SPEC 6.3) |
| `data/latency/*_gpu.jsonl` | 51 | | el mismo subset en GPU, sólo para instrumentar memoria (A17) |
| `data/latency/*_resources.json` | | | pico de RSS y de VRAM de cada una de esas corridas |

Los manifests son append-only y vale la última fila de cada id: las 1086 líneas de Sintética son sus
540 celdas medidas dos veces, y las métricas ven 540.

Las 544 líneas de XTTS no son un error: la celda `slr61_f_01/cor_001/s1234` tiene la fila fallida
del bug de `slice indices` del 2026-08-20 **y** su reintento exitoso 11 minutos después. Es el
diseño funcionando, y el intento fallido queda como registro.

**Sin errores duros en ninguno de los cuatro.** Sintética tuvo 12 con el fork 0.1.9 —`pro_004` en
las cuatro voces por tres seeds, un `'.'` suelto del segmentador que hacía explotar a FastPitch— y
el arreglo llegó en 0.1.10, que es la versión medida acá: las 540 celdas salen y `pro_004` da WER
0.000. El hallazgo queda en SPEC 6.5 porque es lo único que este benchmark le aportó al producto.

Números de referencia para leer todo lo de abajo:

| | XTTS-v2 | Qwen3-TTS | Chatterbox v3 | Sintética (fuera de competencia) |
|---|---|---|---|---|
| tasa de fallo, con `trailing_audio` | 14.6 % | **0.0 %** | 47.9 % | **0.0 %** |
| **la misma, sin esa regla** | **7.9 %** | **0.0 %** | **3.4 %** | **0.0 %** |
| de eso, error duro | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| `boundary_disagreement` (no es falla) | 0.6 % | 0.2 % | 0.4 % | 4.3 % |
| `trailing_disagreement` (no es falla) | 0.0 % | 0.2 % | 0.2 % | 0.6 % |
| WER medio | 0.106 | 0.012 | 0.108 | 0.060 |
| similitud escalada (piso→techo) | 0.642 | 0.848 | 0.815 | **0.601**, y es una cota (A15) |
| RTF p50, GPU | 0.252 | 0.910 | **0.617** (con reinicio cada 27, ver A14) | **0.009** |
| RTF p50, CPU 4 hilos | 2.485 | 5.393 | 5.578 | **0.047** |
| VRAM pico reservada | 2750 MiB | **5166 MiB** | 3880 MiB | **318 MiB** |
| habla pedida p50 | 2.48 s | 2.08 s | 2.07 s | 1.92 s |
| cola p50 / p95 | 0.18 / 1.10 s | 0.16 / 0.34 s | 0.30 / 2.40 s | **0.16 / 0.32 s** |
| no-habla p50 | 0.38 s | 0.60 s | 1.48 s | 0.51 s |
| segundos de referencia | 6.0 | 6.5–11.6 | 6.0 | ninguno: elige la voz por id |

**Sintética se midió dos veces**, con el fork 0.1.9 (2026-09-14) y con el 0.1.10 (2026-09-16), que
arregla el crash de `pro_004`. Los números de arriba son los del 0.1.10. De las 45 frases, **40
salen idénticas byte a byte** entre las dos versiones; cambian las cuatro de `normalizacion`, por el
arreglo del regex de números, y aparece `pro_004`. Ninguna conclusión se movió.

**Las dos primeras filas son el titular del análisis y están en A13.** La regla domina la tasa de
fallo de Chatterbox y la invierte: con la regla es el peor modelo por lejos, sin la regla es el
mejor de los tres después de Qwen. Medido: la regla **acierta** —la cola es audible en el 100 % de
las filas marcadas, y escuchadas las 266, 265 tienen algo adentro— y lo que estaba mal era el nombre
y la contabilidad, las dos cosas ya corregidas. Y la composición dice que son **dos defectos
distintos**: la cola de XTTS es habla inventada en el 59 % de sus filas, la de Chatterbox es ruido
en el 97 %. Lo único que falta para publicar el número es el barrido del umbral al lado.

Escala de similitud, común a los tres: techo 0.827 (0.806–0.852), piso 0.155 (0.085–0.221), techo
corto 0.671 (0.588–0.715).

---

## A1 ★★★ — El desacuerdo de ortografía no es alucinación, y tampoco es el modelo acertando

**Pregunta.** ¿Cuántas filas están clasificadas como falla del modelo cuando lo único que hay es
desacuerdo sobre dónde termina una palabra?

**El caso que lo destapó.** `qwen3_tts__slr61_m_02__yei_002__s9012__ref0`: la frase es "Llamalo ya,
que la llave está en el llavero" y el transcript dice *"Y amalo ya que la llave está en el
llavero"*. Quedó etiquetado **alucinación** —WER 0.222, CER 0.048, una inserción— porque el
alineamiento ve material extra donde el ASR partió una palabra en dos.

**Escuchado el 2026-09-04, y da vuelta la interpretación.** La versión anterior de esta entrada
decía que Whisper estaba escribiendo un yeísmo rehilado bien hecho y nosotros lo contábamos como
error. **No es eso.** En el audio, "Llamalo" **no** se realiza con rehilamiento: es una aproximante,
y "Y amalo" es exactamente lo que Whisper escribe cuando oye [j]. Las otras `ll` de la frase sí
tienen rehilamiento, pero débil para lo que se esperaría de un hablante rioplatense.

O sea que la fila no es un falso positivo que perjudica al modelo: es **una falla dialectal real con
la etiqueta equivocada**. El signo del error estaba al revés, pero al revés de como esta entrada
creía.

**La regla propuesta sigue en pie, sin cambios.** Cuando el WER muestra inserciones pero el CER es
muy bajo, no hay material extra: hay desacuerdo sobre dónde termina una palabra. Pegando todo sin
espacios, `llamaloyaquelallave…` contra `yamaloyaquelallave…` son 2 caracteres de 32 — **CER 0.05
contra WER 0.22**. La divergencia entre las dos medidas *es* la señal, las dos ya se calculan, y
llamar a eso alucinación está mal **independientemente de cómo se haya pronunciado**.

**Lo que cambió es a dónde van las filas reclasificadas: a revisión, no a limpio.** Aplicando el
arreglo como estaba escrito, estas filas quedaban sin falla. Hoy están marcadas —mal— pero marcadas;
después del arreglo, las únicas filas donde el instrumento por accidente ve el fenómeno dialectal se
volverían **invisibles** hasta que exista la métrica de acento de SPEC 7.4. Así que la reclasificación
necesita un veredicto propio, que no es falla ni es éxito. Se llama **`boundary_disagreement`** y
precede a `hallucination` en la cadena de precedencia, que es donde esas filas caen hoy (SPEC 7.5).
Su lista de escucha sale de `rtb.metrics.listening`, con vocabulario cerrado.

**Lo que Whisper escribe, medido.** Sobre los 324 onsets `<ll>` marcados como `target` en
`yeismo_rehilado`, los tres modelos juntos:

| ortografía | n | qué sugiere |
|---|---|---|
| `ll…` ("llamalo", "llámalo") | 315 (97.2 %) | **nada**: la ortografía española es yeísta y cubre [ʃ], [ʒ] y [ʝ] por igual |
| `y…` con espacio ("Y amalo") | 1 (0.3 %) | aproximante [j]/[ʝ] — el caso de arriba |
| `sh/ch…` ("Chovió") | 1 (0.3 %) | rehilamiento fuerte, tanto que Whisper eligió "ch" |
| el target no aparece | 7 (2.2 %) | otros errores; uno es "**yanda**" por "llanta" (`chatterbox_v3__slr61_f_01__yei_003__s1234__ref0`) |

**Y no es sólo el yeísmo: la misma ceguera aparece en `lexico_local`, con la acentuación.**
`chatterbox_v3__slr61_m_02__lex_002__s1234` dice "bondi" —la palabra entera, los fonemas
correctos— **con el acento en la sílaba equivocada**, y esa es plausiblemente la razón de que
Whisper la partiera en "buen día": "bondí" se parece mucho más a "buen día" que "bóndi". Quedó
etiquetada `dijo_mal` (2026-09-15).

Lo que importa no es la fila, es **cómo la agarramos: de casualidad.** Si el ASR hubiera escrito
"bondi" igual, el WER sería 0.000 y el error no existiría para el instrumento. O sea que la
acentuación del léxico local es un segundo fenómeno dialectal que v0 no mide, de la misma familia
que el yeísmo: **acentuar mal una palabra que sí se conoce es exactamente el defecto que
`lexico_local` existe para exponer**, y la categoría lo mide sólo por inteligibilidad. Dos
categorías centrales ciegas a su propio fenómeno es el argumento más fuerte a favor de 7.4.

Un límite del vocabulario que esto deja a la vista, y se declara: `dijo_mal` agrupa "dijo otra cosa"
—"GPU" → "GEP UI"— con "dijo lo correcto mal acentuado", que no son el mismo defecto. No se abre un
valor nuevo a mitad de anotación, porque cambiaría el significado de lo ya escrito; si 7.4 llega, la
distinción vive ahí y no acá.

De ahí sale un subproducto gratis: **la ortografía del ASR es evidencia asimétrica** sobre la
realización. `sh/ch` es señal fuerte de rehilamiento, `y…` es señal de aproximante, y `ll…` no dice
nada. No mide el fenómeno —eso es 7.4— pero señala los extremos sin costo, y sirve para elegir qué
filas escuchar primero.

**El límite que esto destapa, y es más grande que A1.** Si el 97.2 % vuelve como `ll…`, y aun ahí la
realización puede estar floja para rioplatense, entonces **el WER de `yeismo_rehilado` es casi ciego
al yeísmo**: un modelo puede hacer [ʝ] castellano en toda la frase y sacar WER 0.000. Los números
actuales de la categoría no dicen nada sobre el dialecto, dicen que se entendió. Eso va declarado en
el README —la categoría central del proyecto se mide, en v0, sólo por inteligibilidad— y es el
argumento más concreto a favor de que 7.4 no es opcional.

**Qué necesita.** La regla: nada nuevo, `failures.jsonl` y `transcripts.jsonl`, es core, sin GPU, y
recalcular cuesta milisegundos. El nombre está decidido, así que queda implementar el veredicto y
generar su lista.
Y **escucha** para validar que la ortografía de Whisper indique de verdad la realización: hasta
ahora es un clip escuchado una vez más dos casos que apuntan al mismo lado, no una relación medida.

**Resultados posibles.**
- Si aparecen pocas filas → se reclasifican a revisión, la tasa de fallo baja un poco y el conteo de
  filas dialectalmente sospechosas queda explícito en vez de escondido dentro de `hallucinated`.
- Si aparecen muchas en `yeismo_rehilado` → **la categoría entera venía midiendo mal**, y hay que
  revisar qué decía el WER de esa categoría en el commit de XTTS.
- Medido hoy son 1 de 72 filas en Qwen (1.4 % de la categoría) y 1 de 324 onsets sobre los tres
  modelos. La magnitud es chica; lo que importa es que la etiqueta miente en las dos direcciones.

**Ojo.** El caso contrario ya no es hipotético: "Chovió" es un modelo rehilando fuerte y "Y amalo" es
uno que no rehila, y los dos producen divergencia WER-CER. La regla de acá **no** los distingue y no
debe pretenderlo: sólo deja de llamarlos alucinación y los manda a revisar. Distinguir [ʃ] de [ʝ] es
la métrica de acento de SPEC 7.4 (v1).

**Implementado y medido el 2026-09-11, y la regla no quedó como estaba escrita acá.** Un corte de
CER **no** separa los dos casos: en una frase larga una inserción genuina da CER bajísimo. La fila
de CER más bajo de toda la categoría `hallucination` (0.011) es
`chatterbox_v3__slr61_f_02__lar_002__s5678`, donde el ASR metió un "y" entre dos palabras intactas.
Inserción real, no frontera, y cualquier umbral de CER que atrape el caso de Qwen se la lleva
puesta.

**Lo que sí separa es la sustitución.** Partir una palabra *consume* la adyacente además de insertar
otra; una alucinación sólo agrega. La regla quedó entonces como mecanismo y un umbral:
inserción ≥ 1 **y** sustitución ≥ 1 **y** el texto pegado sin espacios difiere por debajo de
`boundary_max_glued_rate`. Sobre los datos separa limpio:

| | ins | sub | qué es |
|---|---|---|---|
| "usuario lo" → "usuario **y** lo" | 1 | 0 | inserción |
| "gpu" → "gpu **puta**" | 1 | 0 | inserción |
| "llamalo" → "y amalo" | 1 | 1 | frontera |
| "retiro" → "de tiros" | 1 | 1 | frontera |
| "bondi" → "buen día" | 1 | 1 | frontera |

**El umbral está sobre una meseta, y la meseta es corta.** Entre 0.08 y 0.10 las filas marcadas no
se mueven. A 0.105 entra material genuino ("cuando termines" → "cuando termine spice") y a 0.179
queda **afuera** una partición real —`chatterbox_v3__slr61_m_01__yei_006__s5678`, "llevate" → "y
bate", y encima de `yeismo_rehilado`—. El corte elegido, 0.10, es conservador y se sabe que pierde
al menos una. El barrido se publica al lado, como el de la cola.

**Movido a 0.12 el 2026-09-14, y esta vez con oído de por medio.** Sintética trajo diez filas nuevas
del mismo caso —"bondi" → "buen día", 9 suyas y 1 de Chatterbox— que caían como alucinación. Se
escucharon: **los modelos dicen "bondi"**. El ASR parte donde no va porque la "a" de "hasta" se le
pega al "di" del final. Perdían **por un carácter**: la frase pegada tiene 26 y cambian 3, contra
los 2.6 que el corte permitía.

El barrido combinado sobre los cuatro modelos, que es lo que hay que publicar al lado:

| corte | filas como frontera (de 46 donde el mecanismo aplica) |
|---|---|
| 0.08 – 0.105 | 7 |
| 0.11 | 8 |
| **0.12 (en uso)** | **18** |
| 0.15 | 20 |
| 0.18 | 24 |

**El precio que se creía pagar no existió.** Al mover el corte entraba también "cuando termines" →
"cuando termine Spice", que esta entrada daba por material genuino insertado — **y era una
suposición, no una escucha**. Escuchado el 2026-09-15: el modelo dice la frase entera y el ASR
inventó "Spice". Esa fila también sale bien de `hallucinated`, así que **el movimiento no costó
ninguna fila mal clasificada**: las 11 que entraron están las 11 bien.

Queda igual la lección de método, que es la parte que se repite: Spice está **más abajo** que el
caso que había que atrapar —0.1053 contra 0.1154—, así que ningún corte los separaba, y probamos si
el mecanismo los distinguía a nivel carácter —Spice son 4 inserciones puras y cero sustituciones—
sin éxito, porque "GPU" → "GEP UI" también tiene cero sustituciones. Que el umbral no pudiera
separarlos **no era el problema**: el problema era creer saber de qué lado caía cada uno sin
escucharlos.

**La tanda quedó completa el 2026-09-15: 18 de 18.** `dijo_bien` 15, `dijo_mal` 2, `dialectal` 1.
Dos de cada tres desacuerdos de frontera son del instrumento, que es la proporción que justifica que
el veredicto exista — pero un tercio no lo son, que es la que justifica que mande a escuchar en vez
de absolver.

Se eligió el error barato, y la asimetría es la que justificó el veredicto desde el principio:
`boundary_disagreement` **manda a escuchar, no absuelve**. Una fila de más en la lista cuesta un
clip; diez de menos cuestan publicar como alucinación a modelos que dijeron la palabra bien.

**El efecto:**

| | antes (0.10) | ahora (0.12) |
|---|---|---|
| XTTS-v2, alucinación | 7.9 % | 7.7 % |
| Chatterbox v3, alucinación | 3.4 % | 3.2 % |
| **Sintética, alucinación** | **1.9 %** | **0.0 %** |
| Sintética, tasa de fallo | 4.4 % | **2.5 %**, que son sólo los errores duros de `pro_004` |

Sintética queda sin ninguna alucinación: todo lo que le quedaba era este caso.

**El efecto sobre las tasas, con la misma fórmula antes y después:**

| | antes | ahora |
|---|---|---|
| XTTS-v2, con `trailing_audio` | 15.2 % | **14.8 %** |
| XTTS-v2, sin esa regla | 8.5 % | **8.1 %** |
| Qwen3-TTS, con `trailing_audio` | 0.4 % | **0.2 %** |
| Qwen3-TTS, sin esa regla | 0.2 % | **0.0 %** |
| Chatterbox v3, con `trailing_audio` | 48.1 % | **48.1 %** |
| Chatterbox v3, sin esa regla | 4.0 % | **3.8 %** |

Son 2 filas en XTTS-v2 y 1 en cada uno de los otros dos — 0.4 %, 0.2 % y 0.2 %. La magnitud es la
que esta entrada anticipaba: chica.

**Y destraba A3 casi sola.** A Qwen3-TTS le queda **una** fila marcada en todo el corpus,
`qwen3_tts__slr61_m_02__lex_004__s1234`, que es exactamente el caso de A2: el "che" que el ASR pudo
haberse comido. Su tasa de alucinación ya es 0.0 %. Si A2 resuelve que el modelo lo dijo, la tasa de
fallo de Qwen es **0.0 %** y la cota superior del error del clasificador sale de escuchar un clip.

**Escuchados los 4 el 2026-09-13, y la regla se valida en 4 de 4: ninguna es alucinación.** El
veredicto reparte las filas en las tres cosas distintas que la regla, a propósito, no separa:

| fila | lo que escribió el ASR | veredicto | de quién es |
|---|---|---|---|
| `chatterbox_v3` `rot_004` | "Retiro" → "de tiros" | `dijo_bien` | del instrumento |
| `xtts_v2` `ang_002` | "Mandame" → "Y dame" | `dijo_bien` | del instrumento |
| `xtts_v2` `ang_003` | "GPU" → "GEP UI" | `dijo_mal` | del modelo |
| `qwen3_tts` `yei_002` | "Llamalo" → "Y amalo" | `dialectal` | del modelo, en la realización |

Que las cuatro se repartan así **es** el resultado: mandarlas a revisión en vez de a limpio era lo
correcto, porque una de cada dos sí es del modelo y absolverlas las habría escondido.

**Y corrige la evidencia asimétrica de arriba.** `ang_002` es una `y…` que **no** es aproximante:
el modelo dice "Mandame" entero y Whisper lo parte igual. O sea que la `y…` inicial sola no indica
realización —es el patrón genérico con que el ASR parte cualquier palabra—, y de la tabla de
ortografías se sostiene sólo `sh/ch` como señal fuerte. Con 2 casos contra 1 no alcanza para medir
nada; alcanza para no apoyarse en `y…` al elegir qué escuchar.

---

## A2 ★★ — La cola marcada puede ser el ASR que se comió una palabra

**Pregunta.** Cuando la última palabra falta del transcript y hay sonido después de la última
palabra transcripta, ¿el modelo alucinó o Whisper no transcribió?

**El caso.** `qwen3_tts__slr61_m_02__lex_004__s1234__ref0`: "Posta que no tenía idea, che." volvió
como *"posta que no tenía idea."* — falta "che" — y quedó etiquetado **cola** (`trailing_audio`)
por el sonido posterior. Lo más probable es que el modelo haya dicho "che" y el ASR se lo haya comido.

**Por qué importa.** Es la otra mitad de A1: la taxonomía le atribuye al modelo algo que hizo el
instrumento. Y "che" no es una palabra cualquiera acá — es léxico rioplatense en posición final,
justo lo que `lexico_local` mide.

**Qué necesita.** **Escucha humana** de las filas afectadas: es la única forma de saber si el audio
dice "che". Cruzar con `audio_shape.speech_end_s` para acotar la lista antes de escuchar.

**Resultados posibles.**
- Si el modelo lo dijo → es una deleción del ASR, la fila no es falla, y hay que agregar la regla
  "deleción final + sonido posterior = probable falla del instrumento, no del modelo".
- Si no lo dijo → es truncamiento, no cola, y la etiqueta está mal igual.

**CERRADO el 2026-09-13. El modelo dice "che" —no del todo bien, pero se escucha— y el ASR se lo
comió.** Es una deleción del instrumento: la fila no es falla, y con eso Qwen3-TTS queda en **0
fallas sobre 540**.

**La regla quedó implementada, y no como absolución sino como veredicto de revisión.** El patrón
—faltan palabras al final **y** hay sonido justo ahí— es exactamente la contradicción entre
truncamiento y cola: las dos reglas no pueden tener razón a la vez, así que ninguna es la respuesta.
Se llama **`trailing_disagreement`**, precede a las dos, y sale de `any_failure` igual que
`boundary_disagreement` (SPEC 7.5). El piso es 1 palabra faltante y **no** `trailing_deletions_min`,
que vale 2: la palabra final suelta es justo el caso que ese umbral evita llamar truncamiento —el
comentario del threshold ya lo anticipaba— y con sonido al lado deja de ser un descarte para pasar
a ser la pregunta.

**Marca 2 filas en todo el corpus, y son las dos respuestas posibles.** Que no coincidan es lo que
justifica que el veredicto mande a escuchar en vez de decidir:

| fila | lo que escribió el ASR | veredicto | de quién es |
|---|---|---|---|
| `qwen3_tts` `lex_004` | "…no tenía idea, che." → "…no tenía idea." | `dijo_bien` | del instrumento |
| `chatterbox_v3` `nor_002` | "del 3,5 %" → "del 3,5" | `dijo_mal` | del modelo |

La segunda apareció escuchando: el modelo dice **"tres com por cinco"**, una mala pronunciación del
numeral que el ASR comprime a "3,5". No es truncamiento —que es como venía contada— ni es cola. Es
la falla que `normalizacion` querría medir, y que A6 explica por qué hoy no se mide.

**El efecto sobre las tasas:** Qwen3-TTS 0.2 % → **0.0 %** y Chatterbox v3 48.1 % → **47.9 %**. XTTS-v2
no tiene ninguna fila con este patrón y queda igual.

---

## A3 ★★★ — Piso de ruido del instrumento

**Pregunta.** ¿Cuál es la tasa de error de la propia taxonomía, y por debajo de qué tasa de fallo
deja de tener sentido publicar una diferencia entre modelos?

**Por qué importa.** Con XTTS en 73 fallas, dos mal clasificadas no cambiaban nada. Con Qwen en
**2 fallas, y las dos sospechosas (A1 y A2)**, el ruido de la taxonomía es del tamaño de la señal.
La tasa real de Qwen podría ser 0.0 %. Publicar "0.4 % contra 15.2 %" sin declarar esto es
publicar una precisión que no tenemos.

**Qué necesita.** Una vez resueltos A1 y A2: revisar a mano **todas** las filas que queden marcadas
como falla en el modelo con menos fallas (son pocas, por definición), y reportar cuántas
sobreviven. Eso da una cota superior del error del clasificador.

**Resultados posibles.** Sea cual sea, el README tiene que decir la tasa **con** el número de filas
revisadas a mano, igual que el WER se publica con su piso de audio real.

**CERRADO el 2026-09-13. Qwen3-TTS es 0.0 % sobre 540 filas, con las 2 marcadas revisadas a oído y
las 2 mal clasificadas.** Una era A1 (`yei_002`, desacuerdo de frontera con realización floja) y la
otra A2 (`lex_004`, deleción del ASR). Las dos salieron del ranking a revisión, ninguna por
absolución.

**La cota, con dos medidas independientes que dan lo mismo:**

| qué mide | número |
|---|---|
| filas mal atribuidas en el modelo con menos fallas | 2 de 540 = **0.4 %** |
| precisión de la regla de cola sobre las filas que marca | 265 de 266 = **99.6 %** |

O sea que **por debajo de medio punto porcentual el instrumento no distingue**, y una diferencia
entre modelos de ese tamaño no se publica como diferencia. Sobre los números de v0 eso no molesta a
nadie: 0.0 contra 14.8 contra 47.9 está tres órdenes de magnitud arriba del ruido.

**Lo que queda sin medir, y se declara:** esto acota los **falsos positivos**, no los falsos
negativos. Nadie escuchó las 538 filas que Qwen pasó limpias, así que la afirmación es "de lo que la
taxonomía marca, esto es lo que se sostiene", no "no hay ninguna falla". Medir lo otro es escuchar
un muestreo aleatorio de filas no marcadas, y no está hecho.

**Y el 2026-09-16 apareció un falso negativo, el primero medido.** Escuchando las 12 filas de
`nor_005` de Sintética, `slr61_m_02` pronuncia **un solo "1"** donde van dos. Es un error real del
modelo, y la taxonomía **no lo cuenta**: cae en `boundary_disagreement`, que sale del ranking por
diseño porque el instrumento no puede decidir de quién es la partición. El oído sí decidió, pero
las escuchas se usan para elegir reglas, no como capa de corrección fila por fila.

Eso obliga a una frase en el README que no estaba prevista: **"0.0 % de fallo" significa "ninguna
falla que la taxonomía automática cuente", no "no se encontraron errores".** En el único modelo con
0.0 % que además se escuchó entero, la escucha encontró uno. Un caso sobre 540 es del orden del
piso que esta entrada mide, así que no invalida el número — pero lo califica, y sin la calificación
el número dice más de lo que se midió.

---

## A4 ★★ — ¿Qwen gana en similitud por clonar mejor o por más segundos de referencia?

**Pregunta.** Qwen recibió 6.5–11.6 s de referencia y XTTS 6.0 exactos (SPEC 6.4). ¿Cuánto de la
ventaja de 0.848 contra 0.642 es eso?

**Lo que ya se midió, y por qué no alcanza.** Dentro de Qwen, `Pearson(ref_seconds_used, cosine) =
−0.115`, y a duración pareja de síntesis Qwen lidera en las seis bandas (+0.05 a +0.18). Pero hay
**sólo cuatro largos de referencia, uno por hablante**, así que el largo está perfectamente
confundido con identidad del hablante. El caso que lo muestra: `m_01` tiene la referencia más corta
(6.55 s) y la similitud más alta (0.780) — pero XTTS **también** clona `m_01` mejor que a los otros
tres (0.713) con 6.0 s parejos, así que `m_01` puede ser simplemente un hablante fácil.

**El chequeo, en dos pasos y el primero es mucho más barato.**

1. **Sobre XTTS, sin construir nada.** XTTS recorta sólo audio, sin texto que alinear: correr con
   `max_ref_seconds` en 3, 6 y 10 sobre los cuatro hablantes mide la pendiente directamente.
   Necesita un config aparte y su propio manifest, o los `generation_id` colisionan y el runner
   saltea todo.
2. **Sobre Qwen, confirmatorio.** Referencias cortadas en borde de utterance para mantener el par
   (audio, texto) alineado. Posible en **3 de los 4** hablantes: f_01 11.62→5.71 s, f_02 8.20→5.82
   o 3.29 s, m_02 7.99→3.89 s. `m_01` es una única utterance de 6.55 s y no se puede partir — justo
   el hablante que motiva la duda.

**Costo.** Paso 1, unos minutos de GPU. Paso 2, 270 síntesis ≈ 12 minutos más construir 3 clips.

**Resultados posibles.**
- **Plana** → la asimetría es inocua, afirmado con una medición *dentro* de hablante en vez de una
  correlación entre hablantes. Cae la salvedad más grande sobre el número principal de similitud.
- **Con pendiente** → sabemos la magnitud y el README la declara con un número.

**Premio lateral.** El protocolo fijó 6.0 s de presupuesto **sin medir nunca si importaba**. Esto lo
contesta, y es información para v1 y para cualquiera que reproduzca el benchmark con otros clips.

**Contestada en gran parte al sumar Chatterbox v3 (2026-08-31).** Chatterbox recibe **6.0 s
exactos**, igual que XTTS-v2, porque su adapter recorta la referencia antes de pasarla (SPEC 6.4).
Con eso hay dos modelos con la misma referencia y uno con más:

| | referencia | similitud escalada |
|---|---|---|
| XTTS-v2 | 6.0 s | 0.642 |
| **Chatterbox v3** | **6.0 s** | **0.815** |
| Qwen3-TTS | 6.5–11.6 s | 0.848 |

Chatterbox llega a 0.815 **con la misma referencia que XTTS**, o sea que casi toda la distancia
hasta Qwen (0.848) se recorre sin un segundo extra de audio. El largo de la referencia puede
explicar, como mucho, los 0.033 que separan a Chatterbox de Qwen — y ni eso, porque son
arquitecturas distintas. La ventaja de Qwen **no** viene de haber recibido más referencia.

Queda abierto sólo el residuo: cuánto de esos 0.033 es referencia y cuánto es modelo. Para eso
todavía sirve el chequeo de abajo, pero ya no es una salvedad sobre el número principal.

---

## A5 ★★ — Qwen supera el techo de similitud en 9.4 % de las filas

**Pregunta.** ¿Por qué 51 de 540 síntesis de Qwen tienen coseno **por encima del techo** (0.827),
contra 3 de 540 en XTTS? ¿Está el modo ICL copiando canal de grabación además de voz?

**CERRADO el 2026-09-18. Copia del prompt hay, y es grande; lo que no hay es nada propio de Qwen:
los tres modelos la tienen y casi en la misma medida.** Medido sobre las 3780 filas contra el otro
clip del mismo hablante —el que la síntesis nunca escuchó—, en CPU y sin regenerar audio.

| | gana por medirse contra su propio prompt | en escala | signo sostenido |
|---|---|---|---|
| XTTS-v2 | +0.027 | +0.040 | 94.1 % de 540 celdas |
| Qwen3-TTS | +0.032 | +0.048 | 92.0 % |
| Chatterbox v3 | +0.035 | +0.053 | 96.3 % |

Es la interacción del cuadrado de A10: lo que queda después de descontar qué clip se usó de prompt y
cuál se usó de objetivo. **Si un modelo copiara sólo identidad, esto daría cero**, porque los dos
clips de un hablante dan 0.853–0.927 entre sí, por encima del techo de 0.827. O sea que en el
espacio del embedding son casi el mismo hablante y aun así el modelo se parece más a uno.

**Por qué no es el modo ICL.** La interacción de Qwen queda entre las otras dos, y su ventaja
sobrevive al clip no visto con el mismo orden que en la diagonal: 0.694 contra 0.673 de Chatterbox y
0.565 de XTTS. La hipótesis de los códigos de habla era plausible y no se sostiene: lo que copia es
el prompt, y lo copian los tres.

**Las 51 filas sobre el techo.** Contra el clip no visto, 38 dejan de superarlo —mediana 0.848 →
0.802— pero eso no es una propiedad de ellas: el delta crece con el nivel del coseno en los tres
modelos por igual (quintil más alto: +0.042 en Qwen, +0.044 en XTTS, +0.032 en Chatterbox), así que
pasar el techo es el extremo del mismo fenómeno y no un modo distinto. Trece quedan arriba del techo
incluso contra un clip que el modelo nunca oyó. Y la composición es la de siempre: `frase_larga`
(24) y `normalizacion` (20), las dos categorías de clips largos, que es el factor de duración de C2.

**Qué va al README.** Que la similitud publicada lleva ~0.03 de coseno —0.04 a 0.05 de escala— que
es prompt y no identidad de hablante, medido, y que afecta a los tres modelos casi por igual, así
que el orden entre ellos no depende de eso. **Se declara, no se resta:** clonar desde ese prompt es
la pregunta del benchmark, y restarlo sería inventar un número.

---

## A6 ★★ — `normalizacion` es inmedible con este instrumento

**Pregunta.** ¿Cómo se mide la verbalización de números, fechas, montos y siglas si Whisper las
vuelve a escribir en dígitos?

**Los datos.** XTTS 60/60 filas no puntuables. Qwen 58/60. El pase detecta el conflicto de formato
y las aparta en vez de contarlas como fallas, y por eso la categoría se movió a revisión manual en
SPEC 7.6, con la evidencia. Se probó `initial_prompt` de Whisper y se rechazó: arregla los números
planos, deja las fechas en dígitos y degrada otro contenido ("AFIP" → "Zafip").

**Por qué importa.** Son 60 filas por modelo, un octavo del grid, sin ningún número publicable. Y la
normalización de texto es de lo más útil de saber para elegir un TTS en producción.

**Ideas.** Escucha manual con planilla de formas aceptadas; o un segundo ASR con vocabulario
forzado; o comparar contra las `accepted_alternatives` a nivel fonético en vez de ortográfico. Las
tres son trabajo de v1.

---

## A7 ★ — El relleno es una métrica en sí misma, y Qwen emite el doble

**Pregunta.** ¿Cuánto silencio pega cada modelo alrededor del habla, y cuánto cuesta eso en un
turno conversacional?

**Los datos.** Relleno mediano 0.208 s en XTTS contra 0.440 s en Qwen. En `enunciados_cortos` la
diferencia es más rara: Qwen **habla menos** (0.40 s contra 0.84 s) y **rellena más** (0.380 s
contra 0.219 s), con casos de 0.36 s de habla envueltos en 0.76 s de silencio.

**Por qué importa.** Dos cosas distintas. Una, valida la decisión de dejar el screening de duración
como triage: el 16.7 % de `enunciados_cortos` que marca en Qwen es **relleno, no habla**, y la tasa
de fallo real de esa categoría es 0 %. Dos, el relleno es latencia real que el RTF no captura: 0.91
de RTF más 0.44 s de cola es un turno que tarda más en cerrar de lo que dice el RTF.

**Actualización (2026-09-03, A13).** Parte de lo que se contaba como relleno era cola audible, y
ahora están separados: el no-habla mediano es 0.38 s en XTTS, 0.60 s en Qwen y 1.48 s en Chatterbox,
del cual 0.18 / 0.16 / 0.30 s es cola y el resto silencio. La comparación de `enunciados_cortos`
sigue en pie con las cifras corregidas: Qwen habla 0.22 s donde XTTS habla 0.50 s.

**Qué necesita.** Nada de medir: la anatomía de cuatro tramos ya sale por modelo y por categoría.
Falta decidir cómo se publica: ¿RTF "efectivo" que incluya el no-habla? ¿Columna aparte? Se decide
al escribir el README.

---

## A8 ★ — La variante 0.6B de Qwen, reencuadrada

**Pregunta.** ¿El Qwen chico mantiene la calidad, siendo el único **zero-shot** que correría en
tiempo real en hardware de consumo?

**Corrección de encuadre (2026-09-14):** "el único" era cierto con tres modelos y dejó de serlo.
Sintética corre a RTF 0.043 en CPU y 0.008 en GPU, así que la frase vale sólo entre los zero-shot —
que es igual la comparación que importa acá, porque Sintética no compite en calidad.

**Por qué cambió el encuadre.** `NOTES_qwen3_tts.md` §7.1 lo dejó como "default sugerido: no, infla
la tabla y desenfoca la pregunta dialectal". Con el perfilado en mano el argumento a favor es más
fuerte: la latencia está limitada por **ancho de banda de memoria**, no por cómputo — 1.917 B
parámetros × 2 bytes = 3.83 GB a leer por pasada, contra ~192 GB/s de esta placa. Si es así, 0.6B
debería ser ~3× más rápido, o sea **RTF ~0.3, competitivo con XTTS en esta misma placa**.

La pregunta deja de ser "¿alcanza el modelo chico?" y pasa a ser "¿el único Qwen usable fuera de un
datacenter mantiene la calidad?". Eso ya no es una fila decorativa.

**Qué necesita.** Un config nuevo apuntando a `Qwen3-TTS-12Hz-0.6B-Base` y su manifest propio; el
adapter no cambia. Grid completo, ~15 minutos de GPU al RTF esperado.

**Pendiente previo, barato.** Verificar la hipótesis de ancho de banda contando tokens generados por
segundo de audio. La aritmética predice ~4 pasadas del talker por frame de 12 Hz (48 × 20 ms ≈ 0.96
s por segundo de audio, que es el RTF medido) pero **eso es inferencia, no medición**.

---

## A9 ★ — TTFB real, que hoy no existe

**Pregunta.** ¿Cuánto tarda cada modelo en entregar el primer audio?

**Estado.** Hoy `ttfb_ms == total_ms` en los dos modelos, y es el dato correcto: sin streaming de
generación el primer audio existe cuando termina todo. Pero XTTS **sí** puede streamear
(`inference_stream`), así que su TTFB real es medible y sería mucho más bajo. Qwen no: su
`non_streaming_mode` sólo simula entrada de texto en streaming, no generación (SPEC 6.4).

**Por qué importa.** Para un agente conversacional el TTFB manda más que el RTF, y es la única
métrica donde XTTS podría ganar por un orden de magnitud en vez de por 3.6×.

**Qué necesita.** Un config `xtts_v2_stream` con `streaming: true`, `reference_only: true` y su
**propio manifest** — si comparte manifest, los `generation_id` colisionan y el runner saltea todo.
Un subconjunto acotado alcanza. Las filas quedan marcadas con `streaming: true` y
`LatencySummary.comparable` ya se encarga de no mezclarlas.

**Salvedad para el README.** Un TTFB de streaming contra un TTFB de modo entero no se comparan; hay
que reportar la pareja (modo, TTFB) y no un solo número por modelo.

---

## A10 ★★ — La fila muerta, ya no: sensibilidad al clip de referencia elegido

**Pregunta.** ¿Cambia algo si se usa el segundo clip de referencia del mismo hablante?

**CERRADO el 2026-09-18, en dos pasos.** El paso 1 (2026-09-06) generó y midió las 540 celdas con
`ref_clip_idx=1` de los tres modelos —1620 filas nuevas, cero errores—, así que la tabla ya no tiene
la fila muerta. El paso 2 completó el cuadrado midiendo cada síntesis contra el clip que **no**
escuchó, y contestó lo que el paso 1 no podía: la ganancia del segundo clip viene del prompt.

**Y de paso salió un bug: la receta que documentaba `configs/grid.yaml` no generaba nada.**
`--ref-clip-indices` era un **filtro** sobre la grilla ya construida, y la grilla se arma con `[0]`,
así que pedir el índice 1 seleccionaba cero celdas — sin error y sin aviso. Ahora ese flag
**extiende** la grilla, que es lo que el propio comentario de `grid.yaml` da por sentado al llamar al
clip de referencia "un FACTOR, no un filtro". El cambio no puede alterar ninguna corrida existente:
cuando los índices pedidos ya están en la grilla el comportamiento es idéntico, y sólo difiere en el
caso que antes devolvía cero.

**Por qué importa.** Es una pregunta que el grid tiene cableada desde el principio y que nunca se
contestó. Si cambiar el clip mueve la similitud, todos los números de similitud tienen una barra de
error que hoy no reportamos.

**Costo, y lo que terminó costando.** Estaba planeado como un recorte de 36 síntesis por modelo —3
frases × 4 hablantes × 3 seeds— pero salió barato hacer el grid entero: 540 por modelo, 9 min de GPU
en XTTS, 26 en Qwen y 30 en Chatterbox, más ~2 min de CPU por modelo para el pase de similitud. Con
eso el corte por `ref_clip_idx` del reporte tiene las dos filas completas en vez de una comparación
sobre tres frases.

`ref_clip_idx` es parte del `generation_id` (`…__s1234__ref1`), así que las filas nuevas no colisionan
con las viejas. Igual fueron a manifests aparte —`<modelo>_ref1.jsonl`— por una razón de reporte y no
de ids: metidas en el manifest del modelo, el próximo `python -m rtb.metrics` recalcularía el WER y la
tasa de fallo sobre 1080 filas que mezclan dos clips de referencia, y eso cambiaría números publicados
en silencio. Como el formato es append-only, fusionarlos cuando el reporte sepa cortar por
`ref_clip_idx` es un `cat` sin pérdida.

**Paso 1 — la diagonal, que es lo que bloquea publicar.** Cada síntesis contra el clip que escuchó,
que es lo único que el pase mide hoy: `sim_row` escribe una fila y un coseno, contra el clip que la
fila declara en `ref_clip_idx`. Se compara **celda por celda** contra la corrida principal —mismo
hablante, misma frase, misma seed— y se lee contra la dispersión entre seeds de la misma celda, que
es el piso de ruido. Comparar promedios no sirve: la duración del clip domina la similitud (C2) y
apareando se cancela.

**Resultado del paso 1.** Apareado celda por celda, y leído contra el piso de ruido, que es el rango
entre las tres seeds de una misma frase:

| | coseno p50 ref0 → ref1 | escalado | \|delta\| apareado p50 | p95 | piso de ruido |
|---|---|---|---|---|---|
| XTTS-v2 | 0.586 → 0.592 | 0.642 → 0.650 | 0.047 | 0.147 | 0.065 |
| Qwen3-TTS | 0.725 → 0.746 | 0.848 → 0.879 | 0.039 | 0.165 | 0.056 |
| Chatterbox v3 | 0.703 → 0.735 | 0.815 → 0.862 | 0.060 | 0.196 | 0.079 |

**1. El efecto por celda está por debajo del ruido entre seeds en los tres modelos.** El número
agregado aguanta el cambio de clip, que es lo que la pregunta pedía saber. Pero el p95 por celda va
de 0.147 a 0.196 en crudo: el agregado es estable y **la fila individual no**, así que dos síntesis
sueltas no se comparan entre clips.

**2. Los tres se corren para el mismo lado: con `ref_01` la similitud sube.** +0.006, +0.021 y
+0.032 en crudo; +0.008, +0.031 y +0.047 en escala. Que tres arquitecturas distintas coincidan en el
signo no es ruido, es un sesgo sistemático, así que **la similitud publicada necesita declarar esta
banda**. El orden entre modelos no se toca —XTTS < Chatterbox < Qwen con los dos clips— y eso
**refuerza A4**: la distancia Chatterbox-Qwen, que ahí se discute como 0.033, con `ref_01` se
achica a 0.017.

Lo que el paso 1 **no** puede decir es *por qué* sube: si `ref_01` es un prompt mejor o simplemente
un objetivo más fácil. Las dos cosas cambian a la vez en la diagonal, y separarlas es exactamente el
paso 2.

**3. La dirección por hablante es una interacción clip-modelo, no una propiedad del clip.** Con dos
modelos parecía lo segundo —`f_01` bajaba en XTTS (−0.040) y en Qwen (−0.026)— pero Chatterbox la
sube (+0.040) y su mayor ganador es `m_01` (+0.050). Con tres modelos la lectura por hablante no se
sostiene.

**4. El clip de referencia mueve cuánta habla produce el modelo, ~6 % en los tres.** Medianas de las
diferencias apareadas: XTTS **+6.2 %** (2.92 → 3.06 s), Qwen **−6.1 %** (2.64 → 2.48 s), Chatterbox
**−6.7 %** (3.60 → 3.36 s). La magnitud es la misma y el signo no. Esto vino de A14, donde explica
que el RTF de Qwen se corriera de 0.910 a 0.996 sin que el modelo fuera más lento —se movió el
divisor— y por eso **el RTF publicado también arrastra una banda por el clip elegido**.

Salvedad que hay que resolver antes de afirmar nada sobre el habla: "produce menos audio" no es
"habla más rápido". Podría ser truncamiento, que sería una falla. Distinguirlos necesita el pase de
ASR sobre las 1620 filas nuevas, unos 15 min de GPU, y de paso daría el WER y la tasa de fallo con
el segundo clip.

**Un detalle que juega a favor del punto 2.** Chatterbox es el que más correlaciona el delta de
coseno con el delta de duración (+0.33) y su audio se acortó. Como la duración correlaciona
*positivamente* con el coseno (C2), el acortamiento debería haber empujado el coseno para abajo — y
subió igual. O sea que el efecto del prompt es todavía mayor que el +0.032 observado. En XTTS pasa lo
inverso y es el más chico de los tres: correlación +0.17 con audio más largo, así que parte de su
+0.006 puede ser duración y no clip.

**Paso 2 — las casillas de afuera de la diagonal. Resultado: `ref_01` es mejor prompt, no objetivo
más fácil.** Las mismas 3780 filas se midieron contra el otro clip del hablante
(`speaker_sim_unseen.jsonl`, `--unseen` del pase de similitud: sin GPU, ~9 min de CPU, sin regenerar
audio), así que el cuadrado de cada celda está completo:

| coseno p50 | desde ref_00, contra ref_00 | desde ref_00, contra ref_01 | desde ref_01, contra ref_00 | desde ref_01, contra ref_01 |
|---|---|---|---|---|
| XTTS-v2 | 0.586 | 0.565 | 0.554 | 0.592 |
| Qwen3-TTS | 0.725 | 0.694 | 0.705 | 0.746 |
| Chatterbox v3 | 0.703 | 0.673 | 0.681 | 0.735 |
| Sintética | 0.559 | 0.547 | — | — |

Con las cuatro casillas, un 2×2 apareado celda por celda separa los tres efectos que el paso 1 veía
mezclados:

| medianas de 540 celdas | copia del prompt | prompt: ref_01 − ref_00 | objetivo: ref_01 − ref_00 |
|---|---|---|---|
| XTTS-v2 | **+0.027** | +0.003 | +0.007 |
| Qwen3-TTS | **+0.032** | +0.010 | +0.004 |
| Chatterbox v3 | **+0.035** | +0.015 | +0.010 |

**1. El efecto del prompt es mayor que el del objetivo en los tres**, y explica la ganancia de la
diagonal del paso 1 (+0.006, +0.021, +0.032): `ref_01` clona mejor, no es más fácil de igualar.

**2. El control que lo cierra por afuera es Sintética.** No escuchó ninguno de los dos clips, así que
su diferencia entre objetivos **es** dificultad del objetivo y nada más — y dice que `ref_01` es
**más difícil** (−0.010), o sea lo contrario de lo que haría falta para explicar el paso 1 por el
lado del objetivo. Es la única fila de la tabla que puede contestar eso, y salió gratis en el mismo
pase.

**3. Lo grande de la tabla no es ninguno de los dos, es la copia del prompt**, de dos a diez veces el
efecto del prompt y presente en los tres modelos. Eso es A5, que cierra con estos mismos números: la
similitud publicada lleva 0.04–0.05 de escala que es prompt y no identidad.

**4. La vara ya estaba medida y por eso el número se puede leer.** Los dos clips de referencia de un
mismo hablante dan coseno 0.880 entre sí (`ref_vs_ref`, C1), por encima del techo de 0.827: en el
espacio del embedding son casi intercambiables, así que una fila que copiara sólo identidad tendría
que dar casi lo mismo contra los dos. Que no dé lo mismo **es** la señal.

**Lo que A10 no es: un test de duración.** Los dos clips de cada hablante miden casi lo mismo
—11.62/10.64, 8.20/8.55, 6.55/6.54, 7.99/7.65— y para XTTS y Chatterbox se recortan a 6.0 s de todas
formas. Lo que varía es *qué* seis segundos de habla, no cuántos; la pendiente por duración es A4 y
se leen separados aunque se corran juntos. `slr61_m_01` es el mejor hablante para esto, con 6.55
contra 6.54 s, y es justo el que motiva la duda de A4.

---

## A11 ★★ — El umbral de enunciados cortos no es publicable

**Pregunta.** ¿Dónde va el techo absoluto de duración de `enunciados_cortos`?

**Estado.** Medido y documentado: sobre la duración del archivo el barrido movía la tasa de **85 %
a 15 %** según el valor, o sea que el número publicado era el umbral y no el modelo. El de ratio, en
cambio, **sí está en una meseta y es publicable**: de 2.5 a 4.0 se mueven 4 filas de 492 en XTTS-v2
—7, 6, 5 y 3, todas 1 % redondeando— y los otros dos modelos no pasan de 1 fila en todo el rango.
Sólo el extremo permisivo del barrido, 2.0, se despega (13 filas, 2.6 %). Duplicar el corte adentro
de la meseta no cambia la respuesta, y eso es lo que hace que el número sea del modelo y no de quien
lo eligió.

**Mejoró al juzgar el habla pedida (2026-09-03, A13).** Con la cantidad corregida el mismo barrido
del margen absoluto (0.3 s → 1.5 s) da:

| | 0.3 s | 0.6 s (en uso) | 1.0 s | 1.5 s |
|---|---|---|---|---|
| XTTS-v2 | 38 % | 27 % | 10 % | 2 % |
| Qwen3-TTS | 6 % | 0 % | 0 % | 0 % |
| Chatterbox v3 | 27 % | 17 % | 8 % | 4 % |

Sigue sin ser una meseta en XTTS, así que el umbral todavía no se publica solo. Pero Qwen queda en
0 % en casi todo el rango, que es una afirmación robusta y no la tenía: **no produce enunciados
cortos del largo equivocado con ningún umbral razonable**.

**Por qué importa menos de lo que parece.** Sólo afecta al **screening de duración**, que es triage
y no la tasa publicada. La taxonomía de fallas no lo usa. Pero mientras el umbral no se calibre, el
screening no se puede mostrar como resultado.

**Qué necesita.** Escuchar los `calibration_clips` reales (0.75–1.17 s) y las síntesis cortas de los
tres modelos, y elegir el techo a oído una sola vez. La cantidad sobre la que hay que calibrar ya
está fija —el habla pedida, no la duración total— así que el número ajustado ahora no hay que
reajustarlo, que era la razón para postergarlo.

Son **144 síntesis** —4 frases × 4 hablantes × 3 seeds × 3 modelos—, ordenadas por habla pedida
creciente, con los 4 clips reales al lado como vara. El veredicto no es por clip: es el punto de la
lista donde el enunciado deja de sonar natural y empieza a sonar cortado. Dos advertencias de
método: la vara **no es pareja** —los clips reales duran lo mismo pero dicen otra cosa, no hay un
"Dale." real en el corpus—, y acá el modelo a la vista **sí** puede sesgar, porque la pregunta es
"¿suena bien?" y no "¿qué hay adentro?". Está más cerca de los juicios de SPEC 7.6 que del resto de
las tandas. Además la lista no sale de la herramienta actual: `rtb.metrics.listening` construye las
tandas desde un campo booleano de `failures.jsonl`, y esto vive en `duration_screen.jsonl`, que es
triage y no se commitea.

**DIFERIDO el 2026-09-13, a después del paso 8.** No bloquea nada: el screening no alimenta ninguna
tasa publicada —la taxonomía de fallas no lo importa— y sin calibrar igual hace su trabajo, que es
detectar una corrida descarrilada antes de gastar una hora de ASR; ésas dan ratios de 5× u 8×, no
1.6 contra 1.4. La mitad publicable del screening, el ratio, ya está sobre su meseta. Y hay un
motivo para esperar además de la falta de urgencia: con Sintética generada la escucha se hace una
vez sobre cuatro modelos en lugar de dos veces sobre tres. **La condición que sostiene esto** es la
que el SPEC ya fija: el screening de duración no se muestra como resultado en el README.

---

## A12 ★★ — El watermark de Chatterbox está dentro de lo que medimos

**Pregunta.** ¿Cuánto mueve el watermark de Chatterbox el WER y el coseno de similitud?

**El hecho.** `ChatterboxMultilingualTTS.generate` aplica `perth.PerthImplicitWatermarker` al audio
**después** de sintetizar, adentro de la librería. Está pensado para ser imperceptible al oído, pero
modifica la forma de onda, y sobre esa forma de onda corren Whisper y ECAPA. No se desactiva —existe
`perth.DummyWatermarker` y algunos forks lo parchean— porque el watermark es parte de lo que el
modelo entrega y medir otra cosa sería medir un fork. Pero está **dentro** de nuestras mediciones y
los otros dos modelos no lo tienen.

**Por qué importa.** Si el efecto no es cero, Chatterbox arrastra una penalidad que XTTS-v2 y
Qwen3-TTS no pagan, y la comparación tiene un sesgo sistemático de un solo lado. Aunque el efecto
sea despreciable, hay que poder decir el número.

**Qué necesita.** Nada de GPU ni de generación nueva: aplicar el watermarker a los **clips reales de
control**, que ya están medidos sin marcar, y volver a medir. El delta de WER y de coseno sobre el
mismo audio es exactamente el sesgo. Corre en `.venv-chatterbox`, que ya tiene `perth`, más
`.venv-eval` y `.venv-simil` para volver a medir.

**CERRADO el 2026-08-31: el delta es despreciable en las tres dimensiones.** Medido aplicando el
watermarker a los ocho `control_clips` reales, que ya estaban medidos sin marcar:

| dimensión | resultado |
|---|---|
| envolvente de energía (cola) | delta medio **−0.004 s**, máximo +0.016 s |
| pico del clip | se mueve ±0.03 sobre valores de 0.13–0.34 |
| coseno ECAPA, marcado vs. sin marcar | **≥ 0.99271** en los ocho clips |
| costo de latencia dentro de `generate` | **0.4–0.6 %** del total (23–37 ms sobre ~6000) |

Un coseno de 0.993 contra sí mismo, sobre una escala cuyo techo es 0.827 y cuyo piso es 0.155, es
ruido: el watermark no mueve la identidad de hablante que mide 7.2. Y no explica nada de la cola
post-habla de A13, que era la sospecha principal.

**Lo que quedó sin medir, y se declara:** el WER sobre los clips marcados. El intento falló por OOM
de CUDA —la GPU estaba ocupada por otro test— y dejó de ser prioritario cuando la envolvente y el
coseno dieron delta cero. Si alguna vez importa, el método está escrito arriba y son ocho clips.

**Conclusión para el README:** se declara que Chatterbox marca su audio y que se verificó que no
afecta las métricas, con estos números. No se desactiva.

---

## A13 ★★★ — La cola post-habla es un defecto real, y la reportamos mal

**Pregunta.** Chatterbox marca 48.1 % de fallas y 45.9 % de eso es `tail`. ¿Es el modelo o es la
regla?

**Es el modelo. La regla acierta; lo que estaba mal era el nombre y la contabilidad, y las dos ya
están corregidas.**

Primero, el error que casi cometimos, porque el razonamiento es tentador y estaba equivocado. De
las 230 filas de Chatterbox marcadas con cola, **189 —el 82.2 %— tienen WER exactamente 0**. Con
eso se puede concluir que son falsos positivos y que sacar la regla bajaría la tasa de 48.1 % a
4.0 %, cambiando el orden entre modelos. **Dos mediciones lo desarman:**

1. **Nivel.** El frame más alto de la cola supera los −30 dB del pico del clip en **el 100 % de las
   268 filas marcadas** de los tres modelos, con mediana ≈ −12 dB. Ninguna es ruido inaudible de
   cuantización. Hay varias con `nivel_db = 0.0`, o sea que **el momento más fuerte del clip está
   en la cola**.
2. **Oído.** `cor_002_s5678` de `slr61_f_02` tiene habla inventada **y** ruido, las dos cosas en el
   mismo clip. Verificado escuchándolo.

WER 0 y cola audible son hechos independientes y los dos son verdad: el texto salió perfecto *y*
el audio tiene sonido de sobra al final.

**El caso que mejor lo muestra**, y no es de Chatterbox: `xtts_v2__slr61_f_02__lex_001__s5678` tiene
**WER 0.000** sobre "Se armó un quilombo bárbaro en la esquina" —siete palabras, ~2.5 s esperados—
y devuelve un clip de **9.70 s con 7.88 s de cola cuyo pico iguala al del clip entero**. El WER no
ve nada. La cola es la única métrica que lo agarra.

**El defecto, cuantificado. Esto es lo que hoy no reportamos:**

Hay **dos** cantidades y confundirlas es fácil: desde la última palabra hasta el fin del **habla**
(sonido audible de más) y hasta el fin del **archivo** (que incluye el silencio final, y ese no es
un defecto).

| desde la última palabra hasta… | XTTS-v2 | Qwen3-TTS | Chatterbox v3 |
|---|---|---|---|
| **fin del habla**, p50 | 0.18 s | 0.16 s | **0.30 s** |
| fin del habla, p95 | 1.10 s | 0.34 s | **2.40 s** |
| fin del archivo, p50 | 0.37 s | 0.26 s | 1.37 s |
| fin del archivo, p95 | 1.45 s | 0.50 s | 2.62 s |
| filas con > 0.5 s de cola audible | 9.4 % | 0.7 % | **43.3 %** |

(Los p95 de habla salían 1.18 y 2.42 en la medición a mano: ahora los calcula `latency.percentile`,
la misma función que el resto del proyecto, y esa es la cifra que imprime la herramienta.)

La cifra que importa es la primera: **sonido audible después de la última palabra**. En la mediana
Chatterbox casi duplica a los otros, pero lo grave está en la cola de la distribución — p95 de
2.42 s contra 0.34 de Qwen, y 43.3 % de las filas por encima de medio segundo. Para un agente
conversacional eso es latencia que el RTF no captura: un turno que no cierra. Plausiblemente peor
que un WER de 0.05, y hoy sale como un booleano en la tabla de fallas en vez de como los segundos
que es.

**Lo que no se puede decidir sin oído.** Si la cola es habla inventada, ruido, o las dos, no sale
del texto. Transcribir el tramo aislado devuelve las leyendas de Whisper para audio poco claro
—*"Gracias."*, *"¡SUSCRIBETE!"*, *"Adiós."*— y eso dice qué hace Whisper con un fragmento
descontextualizado, **no** que no haya habla: en `cor_002` sí la hay. Por eso la composición va a
escucha humana.

**Hecho el 2026-09-03, y es lo que no dependía de la escucha.** `metrics/anatomy.py` deriva el
cuarto tramo al reportar, uniendo `audio_shape.jsonl` con `transcripts.jsonl` —no es un campo nuevo
en `audio_shape`, que sigue siendo ciego al ASR—, `metrics/sensitivity.py` es el punto de barrido
que ahora comparten las dos tasas, y el veredicto pasó a llamarse `trailing_audio` y a guardar sus
segundos. La tabla de anatomía sale por modelo y por categoría en `python -m rtb.metrics`, y el
barrido del umbral con `--sensitivity`. El detalle de diseño quedó en SPEC 7.5.

**Escuchados los 268 el 2026-09-13, y la respuesta es que hay dos defectos distintos con el mismo
nombre.** Ésta es la composición, y es lo que la entrada no podía contestar sin oído:

| | n | habla inventada | ruido | ambas | nada |
|---|---|---|---|---|---|
| XTTS-v2 | 37 | **21 (56.8 %)** | 15 (40.5 %) | 1 | 0 |
| Chatterbox v3 | 230 | 3 (1.3 %) | **223 (97.0 %)** | 2 | 2 |
| Qwen3-TTS | 1 | 0 | 0 | 0 | 1 |

**La cola de XTTS es alucinación; la de Chatterbox es ruido.** Contando `ambas`, 22 de 37 filas de
XTTS (59 %) tienen palabras que nadie pidió, contra 5 de 230 (2.2 %) en Chatterbox. Son dos fallas
de naturaleza distinta y el README no las puede publicar bajo una sola etiqueta: una es el modelo
inventando contenido, la otra es el vocoder que no corta. Para un agente conversacional la segunda
es latencia y la primera es una respuesta equivocada.

**Tres cosas más que salieron de la escucha:**
- **Cero `corte` en todo el corpus.** Ninguna cola es la frase cortándose. La hipótesis existía y
  quedó descartada de una vez.
- **La regla acierta en 265 de 266** (99.6 %; las otras 2 filas del listado original se fueron a
  `trailing_disagreement`, A2). El único falso positivo es
  `chatterbox_v3__slr61_m_01__pro_004__s9012`, 1.42 s a −11.3 dB sobre "Bueno… a ver… ¿cómo te
  explico?".
- **El nivel separa los dos defectos, no sólo el oído.** Pico de la cola: −5.0 dB mediano cuando hay
  habla, −10.1 dB cuando es ruido. Y la duración va al revés de lo esperado: la cola de ruido es más
  larga (p50 1.52 s) que la de habla (1.10 s), porque las de habla son palabras sueltas y las de
  ruido son colas de decaimiento.

**Decidido el 2026-09-19, y con eso la entrada cierra: la tasa se publica como cifra.** A 0.5 s,
con las dos columnas que ya se usan —con y sin la regla de cola— y el barrido completo debajo. La
alternativa era publicar sólo el barrido, y se descartó porque deja al benchmark sin una cifra
comparable, que es lo que se cita. Lo que el README tiene que llevar al lado, todo ya medido acá:

1. las dos columnas, porque la regla sola invierte el ranking;
2. el barrido de 0.2 a 2.0 s, porque la cifra absoluta depende del corte y el **orden no**;
3. que la cola son dos defectos distintos —alucinación en XTTS, ruido en Chatterbox— y no una
   etiqueta sola;
4. los segundos, que es la cantidad real: p50 y p95 por modelo.

**Lo que cambió al arreglar la contabilidad.** Al estar por encima del umbral de silencio, la cola
entraba en `speech_s` y **no** en `padding_s`:

| medianas, en segundos | habla según `audio_shape` | habla pedida | relleno | no-habla |
|---|---|---|---|---|
| XTTS-v2 | 2.69 | 2.48 | 0.21 | 0.38 |
| Qwen3-TTS | 2.24 | 2.08 | 0.44 | 0.60 |
| Chatterbox v3 | 2.80 | **2.07** | 0.51 | **1.48** |

Corrige una conclusión: parecía que Chatterbox hablaba más largo que Qwen (2.80 contra 2.24).
Descontando la cola **hablan casi igual**, 2.07 contra 2.08.

**El screening de duración venía marcando colas, no duraciones.** Juzgaba `duration_s`, el archivo
entero. Sobre el habla pedida: XTTS 7.0 % → 3.5 %, Qwen 1.7 % → **0.0 %**, Chatterbox 9.6 % →
1.7 %. Qwen es el caso que lo deja claro: **nunca produce habla del largo equivocado**, sus marcas
eran todas cola. O sea que el mismo defecto se contaba dos veces con dos nombres. Aparecen además
`too_short` por primera vez —0.6 %, 1.9 %, 0.9 %—: el relleno venía sosteniendo por encima del piso
a clips cortos. Salvedad: la versión corregida necesita el ASR, así que el screening deja de ser
"manifest-only", y cada fila declara qué cantidad juzgó. Sin transcripts sigue corriendo sobre el
archivo, que es para lo que se construyó.

**El barrido del umbral de cola, que es lo que hay que publicar al lado de la tasa** (porcentaje de
filas clasificables marcadas):

| `tail_after_last_word_s` | XTTS-v2 | Qwen3-TTS | Chatterbox v3 |
|---|---|---|---|
| 0.2 s | 35 % | 24 % | 60 % |
| 0.3 s | 15 % | 5 % | 51 % |
| **0.5 s (en uso)** | **8 %** | **0 %** | **46 %** |
| 0.75 s | 6 % | 0 % | 41 % |
| 1.0 s | 5 % | 0 % | 38 % |
| 1.5 s | 2 % | 0 % | 23 % |
| 2.0 s | 1 % | 0 % | 12 % |

La cifra absoluta depende del umbral —XTTS va de 35 % a 1 % según dónde se corte— así que sola no
se publica. El **orden sobrevive a todo el rango**, y Chatterbox sigue arriba del 12 % incluso
exigiendo dos segundos de cola: eso sí es un resultado, y no depende de ningún número elegido.

**Lo que la regla nueva tiene que seguir agarrando.** `cor_002` produjo **24.16 s de audio para
"¿Cómo?"**, con 21.84 s de habla y transcripción incoherente. Es una falla catastrófica y distinta
de una cola.

**Relación con A2 y A3.** A2 es el caso inverso y sigue abierto: cola marcada donde además faltaba
una palabra ("che"), donde hay que separar la cola del modelo de la deleción del ASR. Y A3 cambia
de signo: el piso de ruido del instrumento **no** es falso positivo por nivel, pero el nombre de la
etiqueta sí induce a error a quien lee la tabla, y eso también hay que contarlo.

---

## A14 ★★★ — El RTF de Chatterbox se degrada durante la corrida

**Pregunta.** ¿Cuál es el RTF de Chatterbox, si durante la corrida pasó de 0.62 a 1.51?

**CERRADO el 2026-09-04. El RTF es 0.617 (p95 0.723), no 0.936.** Medido con reinicio de proceso
cada 27 síntesis sobre el grid completo, y con un control que no deja lugar a dudas: las 540 celdas
devolvieron audio de **duración idéntica** a la corrida publicada —540 de 540, delta 0.000 s—, así
que las dos corridas hicieron exactamente el mismo trabajo y la única diferencia es el reloj.

| | corrida publicada | con reinicio cada 27 |
|---|---|---|
| RTF p50 | 0.936 | **0.617** |
| RTF p95 | 1.290 | **0.723** |
| por cuartos, en orden temporal | 0.707 · 0.820 · 0.991 · 1.200 | 0.628 · 0.612 · 0.618 · 0.611 |
| último cuarto / primero | **1.70×** | **0.97×** |
| wall clock de las 540 | 35.1 min | **29.4 min** |

Delta apareado celda por celda: **−0.307**. Y reiniciar 20 veces sale **más barato** que arrastrar
la degradación, aunque cada reinicio pague la carga del modelo.

**Los datos del diagnóstico.** Las 540 síntesis originales, en orden temporal, van de RTF 0.646 en
el primer tercio del primer hablante a 1.255 en el último tercio del último. Sube monótonamente
dentro de cada hablante y entre hablantes, sin cortes. Las últimas filas hacen `frase_larga` a RTF
1.45–1.52 y un proceso nuevo hace el mismo trabajo a 0.65. XTTS-v2 está plano en 0.252 en los cuatro
cuartos y Qwen3-TTS va de 0.941 a 0.912, así que no es la máquina en general.

**Lo que se descartó, medido.**
- **Leak de VRAM**: `allocated` queda clavado en **3287 MiB** en las cuatro ramas de la sonda, de
  punta a punta. No se retiene memoria viva.
- **Térmica pura**: repitiendo una sola frase 40 veces la temperatura sube 12 °C sin penalidad, y
  los otros dos modelos no derivan.
- **El watermark**: 0.4–0.6 % del total (23–37 ms sobre 6000), ver A12.
- **`torch.cuda.empty_cache()` entre síntesis**: **no sirve.** Deriva 1.12× contra 1.14× de la rama
  sin mitigador, dentro del ruido. Baja el pool reservado de 3896 a 3756 MiB y no cambia el RTF. El
  parámetro quedó en el adapter, en `false`, por si alguna vez hace falta volver a probarlo.

**La sonda que eligió el mitigador** (2026-09-04, 90 celdas por rama: un hablante, las 45 frases,
2 seeds — las diez categorías, o sea el rango completo de 1 a 38 palabras). Tres ramas sobre las
mismas celdas, una por vez para que no compitan por la placa:

| tramo de 30 celdas | 1 (yeísmo/rótica/voseo) | 2 (cortos/léxico/topo) | 3 (normalización/prosodia/**largas**) |
|---|---|---|---|
| corrida publicada | 0.640 | 0.706 | 0.757 |
| un solo proceso | 0.630 | 0.675 | 0.696 |
| un solo proceso + `empty_cache` | 0.628 | 0.673 | 0.692 |
| **proceso nuevo por tramo** | **0.624** | **0.645** | **0.619** |
| delta apareado del reinicio vs publicada | −0.017 | −0.063 | **−0.125** |

**El tramo 3 es el que decide, y por eso el diseño importaba.** Los tres tramos no tienen frases del
mismo largo —el tercero se lleva `frase_larga` y `prosodia`—, así que el ascenso 0.63 → 0.70 podía
ser largo de texto en vez de posición. En proceso nuevo el tramo 3 sale a **0.619**, igual que el
tramo 1 (0.624): **es posición, no texto**. La última fila es la lectura limpia: cuanto más adentro
de la corrida, más recupera el reinicio.

**El mecanismo, más afinado que la hipótesis anterior.** Lo que crece es el pool **reservado**, no
el asignado: 3470 → 3896 MiB en un proceso largo. Pero el tamaño del pool no es la causa — el
tercer chunk de la rama con reinicios llega al **mismo** pico de 3896 MiB y es el tramo más rápido
de todos. Lo que distingue a los dos casos es la historia: uno llegó ahí creciendo a lo largo de 60
síntesis de tamaños distintos, el otro de una. O sea que lo que degrada es el **patrón de
fragmentación acumulado por secuencias de largo variable**, no cuánta memoria hay reservada.

**Corrige además la tabla de RTF por categoría.** En la corrida publicada iba de 0.840 (`yeismo`) a
1.134 (`frase_larga`), y se leía como "las frases largas cuestan más". No era la categoría, era la
posición: `yeismo` va primero en el grid y `frase_larga` última. Con reinicios las categorías se
aplastan entre 0.599 y 0.630, con una sola excepción real —`enunciados_cortos` en 0.701— que sí
tiene motivo: overhead fijo sobre poco audio.

**Qué se publica, y con qué salvedades.**
- **El RTF de Chatterbox es 0.617 y va declarado como "con reinicio cada 27 síntesis"**: es una
  condición de medición, no una propiedad del modelo.
- **La asimetría de protocolo va escrita**: XTTS-v2 y Qwen3-TTS se midieron en un proceso único
  porque no derivan. Que no lo necesiten es lo que mantiene la comparación válida.
- **Es una RTX 4050 de 6141 MiB**, y el pool reservado tocó 4508 MiB en la corrida definitiva. La
  fragmentación puede comportarse distinto en una placa con más aire, así que el número viaja con la
  placa, como todos los de latencia (SPEC 7.3).
- Con esto la tabla cambia de orden: XTTS-v2 0.252, **Chatterbox v3 0.617**, Qwen3-TTS 0.910.

**Dónde viven los datos.** `data/latency/`, separado de `data/manifests/` a propósito: la latencia
de Chatterbox sale de `chatterbox_v3_rtf.jsonl` y sus métricas de calidad de la corrida original, y
un manifest de latencia dentro de `data/manifests/` se leería como un cuarto modelo. Las tres ramas
de la sonda quedan al lado, porque son la evidencia de que `empty_cache` se descartó y de que la
deriva no era largo de texto. `configs/models/chatterbox_v3_lat.yaml` es el config, idéntico al
publicado salvo `name` y `manifest_path`. Nada de esto entra en el reporte de resultados:
`python -m rtb.metrics` recibe `--manifest` explícitos y no globea.

**`LatencySummary.comparable` sigue sin detectar esto.** Chequea hardware y modo de streaming, y la
deriva intra-corrida es una tercera forma de incomparabilidad. Hoy se maneja declarándola; si
alguna vez se automatiza, el chequeo es la pendiente del RTF contra la posición.

**El pendiente, cerrado el 2026-09-05: ni XTTS ni Qwen derivan.** Segunda muestra independiente de
cada uno, generada con el segundo clip de referencia para no repetir celdas, cada modelo en un
proceso continuo y **sin un solo corte**:

| | exposición | cuartos, en orden temporal | último/primero |
|---|---|---|---|
| XTTS-v2 `ref1` | 9.1 min, 540 síntesis | 0.256 · 0.253 · 0.255 · 0.255 | **1.00×** |
| Qwen3-TTS `ref1` | 26.1 min, 540 síntesis | 0.980 · 1.010 · 0.983 · 1.001 | **1.02×** |
| Chatterbox v3 `ref0` | 35.0 min, 540 síntesis | 0.707 · 0.820 · 0.991 · 1.200 | **1.70×** |

Chatterbox es el único que deriva, con dos modelos de control medidos dos veces cada uno. Dato de
higiene que salió de paso: la corrida publicada de XTTS tiene **un** corte de 52 min, y está después
de la primera fila; las otras 542 fueron seguidas. Su planitud original era evidencia válida.

**Y apareció algo que no es deriva y toca al RTF igual.** El p50 de Qwen pasó de 0.910 a 0.996 entre
las dos corridas sin derivar dentro de ninguna. No es el largo de la referencia —`slr61_m_01` usa
clips de 6.55 y 6.54 s y su RTF igual sube +0.090— y tampoco es que el modelo se haya vuelto más
lento: **lo que se movió es el divisor**. Con `ref_01` Qwen produce **6.1 % menos audio para el mismo
texto** mientras el reloj se queda alrededor del 1 %. XTTS lo confirma por el otro lado: su reloj
sube ~7 % y su audio ~6 %, así que su RTF queda plano.

**Ojo con qué mediana se lee, porque acá las dos no coinciden.** Hay dos estadísticos distintos y el
documento usa el segundo en varios lados: la **diferencia de medianas** —la mediana de una corrida
contra la de la otra— y la **mediana de las diferencias** apareadas celda por celda. Sobre el reloj
de Qwen dan resultados de signo opuesto:

| Qwen, medianas | RTF | `total_ms` | `duration_s` |
|---|---|---|---|
| ref0 → ref1 (marginal) | 0.910 → 0.996 | 2443.5 → 2422.6 ms | 2.640 → 2.480 s |
| diferencia de medianas | +0.086 | **−20.9 ms** | −0.160 s |
| mediana de las diferencias apareadas | +0.086 | **+27.4 ms** | −0.160 s |

La apareada es la que corresponde para "¿el mismo trabajo salió más lento?", y es la que se usa acá.
Pero lo que sostiene la conclusión es la **magnitud y no el signo**: por cualquiera de las dos
medidas el reloj se mueve ~1 % y el audio ~6 %, y en el audio las dos coinciden exacto. En XTTS no
hay ambigüedad: las dos medidas coinciden en signo y magnitud.

O sea que **el clip de referencia mueve cuánta habla produce el modelo, y el RTF se corre por el
divisor sin que el modelo sea más lento**. El RTF publicado arrastra una banda de ±6 % que viene
sólo de qué clip se eligió, y eso hay que declararlo junto al número. Salvedad: "produce menos
audio" no es "habla más rápido" — podría ser truncamiento, que es una falla, y distinguirlos
necesita el pase de ASR sobre estas filas. La pregunta pasa a A10, que es donde vive el efecto del
clip de referencia.

---

## A15 ★★ — La similitud de Sintética se mide sobre train, y por eso es una cota

**Pregunta.** ¿Cuánto de la similitud de Sintética es identidad de hablante clonada y cuánto es
haber visto esa grabación en el entrenamiento?

**El estado, que es una decisión y no una duda.** Sintética comparte los cuatro hablantes de SLR61
y su fine-tune usó parte de ese corpus. Los `ref_clips` de `speakers.yaml` no se eligieron con el
criterio de quedar afuera, así que su similitud se mide contra audio que el modelo entrenó. **Se
publica igual, como cota superior y declarada** (SPEC 7.2, decisión del 2026-09-13). El dataset
tiene utterances sin usar en el entrenamiento; encontrarlas es una búsqueda que todavía no se hizo.

**Por qué importa, y por qué no alcanza con decir "las dos filas están sesgadas".** Es tentador leer
la asimetría como pareja —el zero-shot vio el audio en el prompt, Sintética lo vio en el
entrenamiento— y no lo es. El sesgo del zero-shot está **acotado y medido**: cambiar el clip objetivo
mueve la similitud entre +0.008 y +0.047 en escala, con el efecto por celda debajo del ruido entre
seeds (A10). Y clonar desde ese prompt es literalmente la pregunta que el zero-shot contesta. El de
Sintética **no tiene cota**: un fine-tune que memorizó la utterance puede dar cualquier coseno.
Publicar las dos como comparables sería inventar una simetría.

**Qué necesita.** Tres cosas, y la primera es la que no depende de este repo:

1. La lista de utterances de SLR61 que el fine-tune de Sintética **no** vio, para esos cuatro
   hablantes. Sale del lado de Sintética, no de acá.
2. Declararlas en `speakers.yaml` como una lista nueva —`heldout_clips`, al lado de `ref_clips` y
   `control_clips`— con sus `source_utterances`. `build_reference_clips.py` las reconstruye
   determinísticamente y el core ya valida que ningún clip comparta audio con otro del mismo
   hablante.
3. Un coseno contra un objetivo que no es el clip escuchado. **Hecho el 2026-09-18**: `--unseen`
   mide contra el otro clip del hablante y escribe su propia métrica, así que la pieza que falta
   para el held-out es el clip, no el código (A5, A10 paso 2).

Y un cuarto detalle que se olvida fácil: **el techo habría que rehacerlo para ese objetivo**. Hoy es
`ctrl_00` contra `ctrl_01`; contra un held-out el techo correcto es held-out contra control. Son
cuatro pares.

**Resultados posibles.**
- Si la similitud de Sintética sobre train ya es **baja** → la conclusión no necesita el held-out, y
  esta entrada se cierra sin buscarlo. Es el resultado barato y hay que mirarlo primero.
- Si es **alta** → no se sabe nada hasta medir contra el held-out, y la cota es lo único publicable.
- Cuando se mida, la distancia entre las dos es el número que importa: **cuánto de la similitud era
  memorización**. Eso no lo contesta ningún otro experimento del proyecto, y de paso da la primera
  medida de qué tan lejos está un fine-tune de un zero-shot cuando a los dos se les pide generalizar.
- Si el fine-tune usó SLR61 **completo** para esos hablantes → no hay held-out posible, la cota es
  definitiva, y eso se declara y se cierra.

**CERRADO el 2026-09-14 por el resultado barato: la cota ya es la más baja de las cuatro.** Medida
con la ventaja puesta —contra audio que estuvo en su entrenamiento— la similitud escalada de
Sintética es **0.601**, y queda última:

| | escalada | medida contra |
|---|---|---|
| Qwen3-TTS | 0.848 | un clip que escuchó 6 s antes |
| Chatterbox v3 | 0.815 | idem |
| XTTS-v2 | 0.642 | idem |
| **Sintética** | **0.601** | audio que estuvo en su fine-tune |

Buscar las utterances held-out sólo puede **bajar** ese número, porque quitaría la ventaja. O sea
que la pregunta de esta entrada —cuánto de la similitud es memorización— ya no cambia ninguna
conclusión: el fine-tune sobre estos hablantes no clona su voz mejor que un zero-shot que los
escucha por primera vez durante seis segundos. Eso es lo publicable, y se publica con la salvedad
de que 0.609 es un techo y no una medición limpia.

**Lo que queda sin medir, y se declara:** cuánto de esos 0.601 es memorización. Si algún día importa
—por ejemplo para decidir si el fine-tune valió la pena— el método está escrito arriba y las
utterances existen. Hoy no hace falta para publicar v0.

---

## A16 ★★★ — Los tres cambios de regla movieron sólo a Sintética, y siempre para abajo

**Pregunta.** Entre el 2026-09-14 y el 2026-09-16 se cambiaron tres reglas de la taxonomía y la
tasa de fallo de Sintética pasó de 2.5 % a 0.0 %. ¿Las reglas se ajustaron al modelo?

**Por qué importa.** Es la objeción correcta que va a hacer cualquiera que lea el README: cambiamos
las reglas y el modelo de la casa quedó en cero. Si la respuesta no está escrita con los números al
lado, el 0.0 % no vale nada — y da lo mismo que la respuesta sea buena.

**El efecto de cada cambio, medido:**

| | XTTS-v2 | Qwen3-TTS | Chatterbox v3 | Sintética |
|---|---|---|---|---|
| corte de frontera 0.10 → 0.12 (alucinación) | 7.9 → 7.7 % | 0.0 % | 3.4 → 3.2 % | **1.9 → 0.0 %** |
| tercera condición del loop (tasa de fallo) | 14.6 % | 0.0 % | 47.9 % | **2.4 → 0.6 %** |
| precedencia de `trailing_disagreement` (ídem) | 14.6 % | 0.0 % | 47.9 % | **0.6 → 0.0 %** |

Los tres zero-shot se mueven **0.2 puntos en total**, y sólo con el primero. Sintética se mueve
2.5. El patrón es real y no se esconde.

**Por qué pasa, que no es "se ajustó al modelo".** Dos razones estructurales, las dos verificables:

1. **Es el único modelo determinístico.** Un error de una celda le aparece **tres veces**, una por
   seed, con audio idéntico bit a bit. Los otros tres tienen variación entre seeds, así que un caso
   límite raramente cae del mismo lado del umbral en las tres. Las 12 filas que movieron los tres
   cambios son **una frase** —`nor_005`— en cuatro voces por tres seeds.
2. **Sus fallas estaban concentradas en un caso.** Después del primer cambio le quedaba una sola
   frase marcada, así que cualquier regla que la tocara movía toda su tasa. A Chatterbox, con 240
   filas marcadas repartidas, ninguna regla sobre un caso le mueve la aguja.

**Lo que NO se puede decir en defensa, y por eso no se dice:** que el efecto era desconocido al
decidir. En los tres casos las filas de Sintética ya estaban medidas y se sabía qué iba a pasar. Lo
que sostiene cada cambio es otra cosa: un mecanismo nombrado y una escucha, no el resultado.

- **Frontera:** Darío escuchó los clips y los modelos dicen "bondi". Perdían por un carácter.
- **Loop:** la condición que faltaba estaba **escrita en el SPEC desde antes** —"repite y se pasa de
  largo"— y su barrido tiene meseta de 1.1 a 1.5, contra el filo del de frontera.
- **Precedencia:** copia el mecanismo que `boundary_disagreement` ya usaba, y el argumento es
  independiente del modelo: a un transcript al que le faltan 10 palabras del final no se le puede
  contar material de más.

**Qué va al README.** Esta tabla, y la frase de que los tres cambios movieron sólo a un modelo. Sin
eso, el 0.0 % de Sintética es una afirmación que el lector no puede auditar.

**Contraprueba barata que no está hecha:** volver a correr la taxonomía con las reglas del
2026-09-13 y publicar las cuatro tasas viejas al lado de las nuevas. Son segundos de CPU y cierra la
discusión con un número en vez de con un argumento.

---

## A17 ★ — La misma celda corre 2 a 6 % más lento en una corrida corta

**Pregunta.** El subset de costo en GPU dio RTF por encima del grid en los tres modelos. No es qué
celdas se eligieron: apareando celda por celda, **esas mismas 48 celdas** dentro del grid dan la
mediana del grid. ¿Qué cobra la corrida corta?

| | grid completo | esas 48 en el grid | corrida de 48 | delta apareado |
|---|---|---|---|---|
| XTTS-v2 | 0.252 | 0.252 | 0.257 | **+0.005** |
| Qwen3-TTS | 0.910 | 0.919 | 0.984 | **+0.055** |
| Chatterbox v3 | 0.617 | 0.929 | 0.666 | −0.183 |

Chatterbox se lee distinto y por eso no entra en la pregunta: su columna del grid es la corrida
original con la deriva de A14 encima, así que su delta mezcla dos efectos y el único limpio es que
48 síntesis en un proceso derivan menos que 543.

**Por qué importa.** El eje de costo compara un **subset de CPU** contra un **grid de GPU**, así que
la diferencia entra entera en el cociente que el README va a publicar. Y la banda que el SPEC pone
al lado del subset (6.3) no la cubre: esa banda dice qué pasa si te tocan otras celdas, medida
remuestreando el grid, y acá las celdas son las mismas.

**Lo que ya está descartado, medido el 2026-09-18:**
- **No es la selección de celdas.** Es el apareamiento de la tabla: la columna del medio es el grid
  sobre las mismas 48 celdas.
- **No es la máquina ni la temperatura.** XTTS repetido con la placa caliente —56 °C contra 46 °C al
  arrancar— dio 0.258 contra 0.257, y la VRAM idéntica al MiB.
- **No es warm-up faltante.** Las tres corridas hicieron sus 3 síntesis de warm-up, fuera del grid.

**Qué significaría cada resultado.** Si es costo de arranque que se diluye —cachés de kernels,
autotune, el asignador todavía sin fragmentar—, una corrida más larga converge a la cifra del grid y
lo que hay que hacer es declarar el sesgo del subset o medir más celdas. Si **no** converge, es una
propiedad de las sesiones cortas, y entonces es información útil en vez de un artefacto: un
deployment que levanta el modelo por turno paga eso mismo.

**Costo de cerrarlo:** correr el subset de XTTS con N creciente —48, 96, 192— y ver si el delta se
achica. Diez minutos de placa, sin tocar código: es `--phrase-ids` con más frases.

**Qué va al README mientras tanto.** Una línea: el RTF de CPU sale de un subset y el de GPU del
grid, y esa asimetría vale unos puntos porcentuales a favor de la GPU, medidos acá.

---

## A18 ★★ — ¿El 0.601 de Sintética es identidad o es el vocoder?

**Pregunta.** En la escucha se reconoce a la misma persona, pero el audio tiene distorsión. ECAPA
castiga el canal además de la voz —está medido dos veces en este proyecto: el resample equivocado
deja el coseno en 0.173 contra su propio embedding (C5), y cada modelo gana 0.03 por medirse contra
el clip que escuchó (A5)—. Si la distorsión del vocoder empuja el coseno para abajo, el 0.601
mezcla **identidad** con **fidelidad de señal**, y no es comparable con nada.

**Por qué importa, y es en la dirección contraria a A15.** Esa entrada dice que el número es una
cota **superior** porque se mide contra audio de entrenamiento. Ésta dice que además puede estar
subestimado por el vocoder. Las dos a la vez significan que el 0.601 no acota nada hasta que se
sepa cuánto pesa cada efecto, y hoy se publica como la cifra más baja de las cuatro.

**Qué necesita: copy-synthesis, que es un techo nuevo.** Pasar el clip de referencia **real** por el
vocoder de Sintética —sin cambiar de hablante, sin sintetizar texto— y medir ese coseno contra el
original. Es exactamente la misma idea que el techo corto: mismo hablante, mismo contenido, misma
cadena de señal, y lo único que varía es lo que se quiere aislar.

**Costo.** Cuatro clips, uno por hablante, y cuatro cosenos. Minutos, sin reentrenar nada. Lo único
que hay que resolver es invocar el vocoder solo, sin el modelo acústico, en `.venv-sintetica`.

**Qué significaría cada resultado.**
- **Techo ≈ 0.60.** Sintética está *en* el techo de su propio vocoder: el número publicado mide
  fidelidad y no clonado, y su fila necesita esa escala propia al lado o deja de decir lo que
  parece decir.
- **Techo ≈ 0.80**, cerca del techo real de 0.827. El vocoder no es el problema y el 0.601 es
  identidad: se publica como está.
- **Algo intermedio.** Se publica el techo al lado de la fila, que es lo que ya se hace con el techo
  corto para enunciados cortos.

**La salvedad que esta entrada no puede sacarse de encima.** Los tres zero-shot también tienen
vocoder y también pagan lo suyo, así que el control riguroso sería un techo por modelo. No es
factible igual de barato: en Sintética el vocoder es un componente separable y en los otros tres
está enredado con el acústico. Si se mide, se mide para uno y se declara que los otros no.

**De dónde salió:** de escuchar los samples, el 2026-09-19. Ninguna métrica lo insinuó.

---

## Cerrados

Se dejan con su respuesta porque el valor está en no volver a discutirlos.

| # | Pregunta | Respuesta | Dónde quedó |
|---|---|---|---|
| C1 | ¿Los clips de referencia concatenados con 250 ms de silencio dañan la identidad del hablante? | **No.** La referencia concatenada más débil da 0.823 contra un control real del mismo hablante, por encima del par real mismo-hablante más débil (0.806). `ref_vs_ref` 0.880. | SPEC 7.2, `speaker_scale.jsonl` |
| C2 | ¿La similitud por categoría se puede leer directamente? | **No.** Está confundida con la duración del clip: Pearson 0.53 (XTTS) y 0.55 (Qwen), y a duración pareja las categorías se aplastan en 0.05 y 0.10 de dispersión contra 0.41 y 0.49 en la tabla por categoría. | SPEC 7.2, nota calculada por modelo en el reporte |
| C3 | ¿El déficit de `enunciados_cortos` es todo artefacto de duración? | **No.** Contra el techo corto (0.671, audio real de la misma duración) el tramo sub-segundo queda en 0.362 (XTTS) y 0.468 (Qwen) contra 0.642 y 0.848 del conjunto. Hay déficit real encima del sesgo. | SPEC 7.2 |
| C4 | ¿Hace falta normalizar loudness para la similitud? | **No.** ECAPA da coseno exactamente 1.0 sobre el mismo clip escalado −12, −6 y +6 dB. | `envs/speaker_sim.txt` |
| C5 | ¿El resample a 16 kHz es higiene o es la métrica? | **Es la métrica.** El mismo clip a 22050 Hz da coseno 0.173 contra su propio embedding a 16 kHz, apenas por encima del 0.155 que se mide entre hablantes distintos. Sin resample devuelve ruido sin error. | SPEC 7.2, `audio_utils.py` |
| C6 | ¿La latencia de Qwen es un bug del adapter? | **No, es tamaño.** 2082 M parámetros contra 467 M de XTTS: 4.5× los parámetros, 3.8× la latencia. Limitado por ancho de banda de memoria, no por cómputo. | `envs/qwen3_tts.txt` |
| C7 | ¿Se puede recortar la referencia de Qwen a 6 s como XTTS? | **No sin desalinear el par (audio, texto).** El corte por prefijo de utterances da 3.70–6.55 s según hablante y `slr61_m_01` no tiene punto de corte. Se pasa la referencia completa y se registra por fila. | SPEC 6.4 |
| C8 | ¿`initial_prompt` de Whisper arregla `normalizacion`? | **No.** Arregla los números planos, deja las fechas en dígitos y degrada otro contenido ("AFIP" → "Zafip"). Cambiar un artefacto por otro no es un arreglo. | SPEC 7.6 |
