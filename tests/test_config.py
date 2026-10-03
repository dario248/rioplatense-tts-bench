"""Config validation, including the committed configs — the same checks CI runs on every push."""

from __future__ import annotations

from pathlib import Path

import pytest

from rtb.config import (
    DEPRECATED_PHRASE_IDS,
    ConfigError,
    Phrase,
    Speaker,
    categories_covered,
    has_onset_for,
    load_grid,
    load_model_config,
    load_phrases,
    load_speakers,
    words_of,
)
from rtb.schema import Phenomenon, PhraseCategory

YEISMO = PhraseCategory.YEISMO_REHILADO.value


def phrase_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "yei_001",
        "text": "Llovió toda la noche y la calle amaneció inundada.",
        "category": YEISMO,
        "failure_hypothesis": "Realiza [ʝ] en vez de [ʃ].",
        "targets": [{"word": "Llovió", "phenomenon": "yeismo"}],
    }
    base.update(overrides)
    return base


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# ── configs commiteados ──────────────────────────────────────────────────────────────────────


def test_committed_configs_are_valid(repo_root: Path) -> None:
    phrases = load_phrases(repo_root / "configs" / "phrases.yaml")
    speakers = load_speakers(repo_root / "configs" / "speakers.yaml")
    grid = load_grid(repo_root / "configs" / "grid.yaml")
    model = load_model_config(repo_root / "configs" / "models" / "dummy.yaml")

    assert len(phrases) == 45
    assert len(speakers) == 4
    assert len(grid.seeds) == 3
    assert model.name == "dummy"

    # 2F / 2M, como pide el alcance de v0.
    assert sorted(speaker.gender.value for speaker in speakers) == ["F", "F", "M", "M"]


def test_committed_phrases_cover_every_category(repo_root: Path) -> None:
    counts = categories_covered(load_phrases(repo_root / "configs" / "phrases.yaml"))
    empty = [name for name, count in counts.items() if count == 0]
    assert not empty, f"categories with no phrases: {empty}"
    assert len(counts) == 10


def test_committed_target_counts(repo_root: Path) -> None:
    # El presupuesto de mediciones de acento de v0: 20 targets de yeismo + 8 de rotica por
    # hablante y seed, o sea 28 x 4 hablantes x 3 seeds x 3 modelos = 1008 mediciones. Si una
    # frase pierde un target en una edicion, este test lo dice.
    phrases = load_phrases(repo_root / "configs" / "phrases.yaml")
    yeismo = sum(len(p.targets_for(Phenomenon.YEISMO)) for p in phrases)
    rhotic = sum(len(p.targets_for(Phenomenon.RHOTIC)) for p in phrases)
    assert (yeismo, rhotic) == (20, 8)


def test_committed_spoken_length_diverges_only_where_expected(repo_root: Path) -> None:
    # Las metricas de truncamiento dividen por n_spoken_words: si coincidiera siempre con
    # n_words, el campo no haria falta. Divergen exactamente en las frases con digitos.
    phrases = load_phrases(repo_root / "configs" / "phrases.yaml")
    diverging = {p.id for p in phrases if p.n_spoken_words != p.n_words}
    assert diverging == {"nor_001", "nor_002", "nor_003", "nor_004", "nor_005"}
    assert next(p for p in phrases if p.id == "nor_005").n_spoken_words == 24


def test_committed_speakers_declare_their_provenance(repo_root: Path) -> None:
    # Sin `source_speaker` y `source_utterances`, "usamos 4 hablantes de SLR61" no es
    # reproducible: nadie puede reconstruir los clips.
    speakers = load_speakers(repo_root / "configs" / "speakers.yaml")
    for speaker in speakers:
        assert speaker.source == "slr61"
        assert speaker.source_speaker, speaker.id
        assert (len(speaker.ref_clips), len(speaker.control_clips)) == (2, 2)
        assert len(speaker.calibration_clips) == 1
        for clip in speaker.all_clips:
            assert clip.source_utterances, f"{speaker.id}/{clip.path.name}"
            assert clip.duration_s


def test_committed_reference_clips_cover_the_protocol_budget(repo_root: Path) -> None:
    # Los refs tienen que superar el max_ref_seconds mas alto de los modelos, o un hablante
    # entregaria menos segundos de referencia que otro y la similitud deja de ser comparable.
    budget = load_model_config(repo_root / "configs" / "models" / "xtts_v2.yaml").max_ref_seconds
    for speaker in load_speakers(repo_root / "configs" / "speakers.yaml"):
        for clip in speaker.ref_clips:
            assert clip.duration_s and clip.duration_s >= budget, f"{speaker.id}/{clip.path.name}"


def test_committed_first_control_is_content_matched(repo_root: Path) -> None:
    # El piso de similitud compara controles de hablantes distintos: si ctrl_00 no dice lo mismo
    # en los cuatro, el coseno mezcla identidad con contenido y el piso deja de significar eso.
    speakers = load_speakers(repo_root / "configs" / "speakers.yaml")
    transcripts = {speaker.control_clips[0].transcript for speaker in speakers}
    assert len(transcripts) == 1, transcripts


def test_committed_control_clips_carry_exact_transcripts(repo_root: Path) -> None:
    # Son la referencia de las filas de ground truth (SPEC 5.5).
    for speaker in load_speakers(repo_root / "configs" / "speakers.yaml"):
        for clip in [*speaker.control_clips, *speaker.calibration_clips]:
            assert clip.transcript and clip.transcript.strip()


def test_committed_clips_exist_when_the_corpus_is_present(repo_root: Path) -> None:
    # data/refs/ no se commitea, asi que en CI este test se saltea; corriendo local, verifica que
    # el YAML y el disco no se separaron.
    speakers_path = repo_root / "configs" / "speakers.yaml"
    first = load_speakers(speakers_path)[0].ref_clips[0].path
    if not (repo_root / first).exists():
        pytest.skip("reference clips not built in this checkout")
    load_speakers(speakers_path, root=repo_root, check_paths=True)


def test_every_committed_model_config_loads(repo_root: Path) -> None:
    # El pinneo del revision se verifica en test_manifest_provenance.py, que es donde se puede
    # cruzar el config contra el manifest que produjo.
    configs = sorted((repo_root / "configs" / "models").glob("*.yaml"))
    assert configs
    for path in configs:
        assert load_model_config(path).name == path.stem


# ── largo y checksum ─────────────────────────────────────────────────────────────────────────


def test_n_words_is_derived_from_the_text() -> None:
    assert Phrase.model_validate(phrase_dict()).n_words == 9


def test_authored_n_words_acts_as_a_checksum() -> None:
    assert Phrase.model_validate(phrase_dict(n_words=9)).n_words == 9
    with pytest.raises(ValueError, match="n_words"):
        Phrase.model_validate(phrase_dict(n_words=99))


def test_spoken_length_comes_from_the_verbalization() -> None:
    phrase = Phrase.model_validate(
        phrase_dict(
            id="nor_001",
            text="El total es $15.750 con IVA incluido.",
            category=PhraseCategory.NORMALIZACION.value,
            targets=[],
            expected_verbalization=(
                "el total es quince mil setecientos cincuenta pesos con iva incluido"
            ),
        )
    )
    assert phrase.n_words == 7
    assert phrase.n_spoken_words == 11
    assert phrase.spoken_reference.startswith("el total es quince")


def test_spoken_length_falls_back_to_the_text() -> None:
    phrase = Phrase.model_validate(phrase_dict())
    assert phrase.n_spoken_words == phrase.n_words


# ── targets ──────────────────────────────────────────────────────────────────────────────────


def test_target_word_must_appear_in_the_text() -> None:
    with pytest.raises(ValueError, match="does not appear"):
        Phrase.model_validate(phrase_dict(targets=[{"word": "yerba", "phenomenon": "yeismo"}]))


def test_yeismo_target_must_be_word_initial() -> None:
    # "calle" tiene el fonema intervocalico: medirlo exigiria un alineador forzado, que es justo
    # lo que la restriccion de SPEC 5.1 evita.
    with pytest.raises(ValueError, match="word-initial"):
        Phrase.model_validate(phrase_dict(targets=[{"word": "calle", "phenomenon": "yeismo"}]))


def test_rhotic_target_must_be_word_initial() -> None:
    with pytest.raises(ValueError, match="word-initial"):
        Phrase.model_validate(
            phrase_dict(
                id="rot_005",
                text="Pero el perro se quedó en la terraza.",
                category=PhraseCategory.ROTICA.value,
                targets=[{"word": "perro", "phenomenon": "rhotic"}],
            )
        )


def test_phenomenon_and_onset_must_agree() -> None:
    # "Llovió" empieza con ll: es un target de yeismo, no de rotica.
    with pytest.raises(ValueError, match="not measurable as rhotic"):
        Phrase.model_validate(phrase_dict(targets=[{"word": "Llovió", "phenomenon": "rhotic"}]))


def test_the_conjunction_y_is_never_a_target() -> None:
    # La "y" aislada es la vocal [i], no la fricativa.
    with pytest.raises(ValueError, match="not measurable as yeismo"):
        Phrase.model_validate(phrase_dict(targets=[{"word": "y", "phenomenon": "yeismo"}]))


def test_unknown_phenomenon_is_rejected() -> None:
    with pytest.raises(ValueError):
        Phrase.model_validate(phrase_dict(targets=[{"word": "Llovió", "phenomenon": "sibilante"}]))


def test_yeismo_category_requires_a_yeismo_target() -> None:
    with pytest.raises(ValueError, match="at least one yeismo target"):
        Phrase.model_validate(phrase_dict(targets=[]))


def test_a_phrase_may_mix_phenomena() -> None:
    # rot_003 mide tres roticas y un yeismo en la misma frase.
    phrase = Phrase.model_validate(
        phrase_dict(
            id="rot_003",
            text="Recién llegó del río, con la ropa mojada.",
            category=PhraseCategory.ROTICA.value,
            targets=[
                {"word": "Recién", "phenomenon": "rhotic"},
                {"word": "llegó", "phenomenon": "yeismo"},
                {"word": "río", "phenomenon": "rhotic"},
                {"word": "ropa", "phenomenon": "rhotic"},
            ],
        )
    )
    assert len(phrase.targets_for(Phenomenon.RHOTIC)) == 3
    assert len(phrase.targets_for(Phenomenon.YEISMO)) == 1


def test_listen_only_phrases_may_have_no_targets() -> None:
    phrase = Phrase.model_validate(
        phrase_dict(
            id="top_001",
            text="Bajate en Callao y caminá dos cuadras.",
            category=PhraseCategory.TOPONIMIA.value,
            targets=[],
            notes="Fonema intervocálico: escucha manual.",
        )
    )
    assert phrase.targets == []


@pytest.mark.parametrize(
    "word,phenomenon,expected",
    [
        ("llave", Phenomenon.YEISMO, True),
        ("Yo", Phenomenon.YEISMO, True),
        ("y", Phenomenon.YEISMO, False),
        ("calle", Phenomenon.YEISMO, False),
        ("Jujuy", Phenomenon.YEISMO, False),
        ("Retiro", Phenomenon.RHOTIC, True),
        ("río", Phenomenon.RHOTIC, True),
        ("perro", Phenomenon.RHOTIC, False),
    ],
)
def test_onset_detection(word: str, phenomenon: Phenomenon, expected: bool) -> None:
    assert has_onset_for(word, phenomenon) is expected


def test_words_of_strips_punctuation_and_case() -> None:
    assert words_of("¿Sabés si podés venir mañana?") == ["sabés", "si", "podés", "venir", "mañana"]


# ── verbalizacion esperada ───────────────────────────────────────────────────────────────────


def test_normalizacion_requires_expected_verbalization() -> None:
    with pytest.raises(ValueError, match="requires expected_verbalization"):
        Phrase.model_validate(
            phrase_dict(
                id="nor_001",
                text="El total es quince pesos.",
                category=PhraseCategory.NORMALIZACION.value,
                targets=[],
            )
        )


def test_any_phrase_with_digits_requires_expected_verbalization() -> None:
    # El normalizador no expande numeros a palabras: sin verbalizacion, la referencia del CER
    # contendria digitos que ningun hablante dice.
    with pytest.raises(ValueError, match="contains digits"):
        Phrase.model_validate(
            phrase_dict(
                id="lex_009",
                text="Me tomé el 152 hasta el laburo.",
                category=PhraseCategory.LEXICO_LOCAL.value,
                targets=[],
            )
        )


def test_verbalization_must_already_be_normalized() -> None:
    with pytest.raises(ValueError, match="not in normalized form"):
        Phrase.model_validate(
            phrase_dict(
                id="nor_002",
                text="La inflación fue del 3,5%.",
                category=PhraseCategory.NORMALIZACION.value,
                targets=[],
                expected_verbalization="La inflación fue del tres coma cinco por ciento.",
            )
        )


def test_alternatives_must_be_normalized_too() -> None:
    with pytest.raises(ValueError, match="not in normalized form"):
        Phrase.model_validate(
            phrase_dict(
                id="nor_002",
                text="La inflación fue del 3,5%.",
                category=PhraseCategory.NORMALIZACION.value,
                targets=[],
                expected_verbalization="la inflacion fue del tres coma cinco por ciento",
                accepted_alternatives=["la inflación fue del tres punto cinco por ciento"],
            )
        )


def test_alternatives_need_a_primary_verbalization() -> None:
    with pytest.raises(ValueError, match="without expected_verbalization"):
        Phrase.model_validate(
            phrase_dict(accepted_alternatives=["llovio toda la noche"]),
        )


# ── ids y categorias ─────────────────────────────────────────────────────────────────────────


def test_retired_category_is_rejected() -> None:
    # s_implosiva salio de v0: es un fenomeno de registro, no de dialecto, y SLR61 es lectura
    # cuidada, asi que no hay falla que detectar.
    with pytest.raises(ValueError):
        Phrase.model_validate(phrase_dict(category="s_implosiva"))


def test_deprecated_ids_cannot_be_reused(tmp_path: Path) -> None:
    assert "sim_001" in DEPRECATED_PHRASE_IDS
    path = write_yaml(
        tmp_path / "phrases.yaml",
        """
- id: sim_001
  text: "Las costas están más frescas."
  category: lexico_local
  failure_hypothesis: "x"
  targets: []
""",
    )
    with pytest.raises(ConfigError, match="deprecated phrase ids reused"):
        load_phrases(path)


def test_duplicate_phrase_ids_are_rejected(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "phrases.yaml",
        """
- id: vos_001
  text: "Vos tenés razón."
  category: voseo
  failure_hypothesis: "x"
- id: vos_001
  text: "Vos querés algo."
  category: voseo
  failure_hypothesis: "y"
""",
    )
    with pytest.raises(ConfigError, match="duplicate phrase ids"):
        load_phrases(path)


def test_error_message_names_the_file_and_entry(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "phrases.yaml",
        """
- id: vos_001
  text: "Vos tenés razón."
  category: voseo
  failure_hypothesis: "x"
- id: yei_009
  text: "La calle está mojada."
  category: yeismo_rehilado
  failure_hypothesis: "y"
  targets:
    - {word: "calle", phenomenon: yeismo}
""",
    )
    with pytest.raises(ConfigError, match=r"phrases\.yaml: entry #1"):
        load_phrases(path)


# ── hablantes ────────────────────────────────────────────────────────────────────────────────


def speaker_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "slr61_f_01",
        "gender": "F",
        "ref_clips": [{"path": "a.wav"}],
        "control_clips": [{"path": "c1.wav"}, {"path": "c2.wav"}],
    }
    base.update(overrides)
    return base


def test_two_control_clips_are_required() -> None:
    # El techo de similitud es la media entre DOS controles del mismo hablante (SPEC 7.2).
    with pytest.raises(ValueError):
        Speaker.model_validate(speaker_dict(control_clips=[{"path": "c1.wav"}]))


def test_control_clip_cannot_reuse_a_reference() -> None:
    # Medir contra el propio prompt infla la similitud y rompe el techo.
    with pytest.raises(ValueError, match="more than one clip"):
        Speaker.model_validate(
            speaker_dict(control_clips=[{"path": "a.wav"}, {"path": "c2.wav"}])
        )


def test_clips_cannot_share_source_audio() -> None:
    # Dos rutas distintas pueden contener el mismo audio si se construyeron de la misma
    # utterance: eso es lo que de verdad rompe el techo, y la ruta sola no lo detecta.
    with pytest.raises(ValueError, match="clips must not share audio"):
        Speaker.model_validate(
            speaker_dict(
                ref_clips=[{"path": "a.wav", "source_utterances": ["001", "002"]}],
                control_clips=[
                    {"path": "c1.wav", "source_utterances": ["002"]},
                    {"path": "c2.wav", "source_utterances": ["003"]},
                ],
            )
        )


def test_calibration_clips_are_optional_and_counted_as_clips() -> None:
    speaker = Speaker.model_validate(
        speaker_dict(calibration_clips=[{"path": "k1.wav", "duration_s": 0.9}])
    )
    assert len(speaker.all_clips) == 4
    assert Speaker.model_validate(speaker_dict()).calibration_clips == []


def test_check_paths_reports_missing_clips(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "speakers.yaml",
        """
- id: slr61_f_01
  gender: F
  ref_clips:
    - path: data/refs/nope.wav
  control_clips:
    - path: data/refs/c1.wav
    - path: data/refs/c2.wav
""",
    )
    load_speakers(path)  # sin chequeo de rutas: valido, es el caso de CI
    with pytest.raises(ConfigError, match="clip files not found"):
        load_speakers(path, root=tmp_path, check_paths=True)


# ── grilla y modelo ──────────────────────────────────────────────────────────────────────────


def test_duplicate_seeds_are_rejected(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "grid.yaml", "seeds: [1, 1, 2]\n")
    with pytest.raises(ConfigError, match="duplicate"):
        load_grid(path)


def test_grid_defaults_to_the_first_reference_clip(tmp_path: Path) -> None:
    grid = load_grid(write_yaml(tmp_path / "grid.yaml", "seeds: [1]\n"))
    assert grid.ref_clip_indices == [0]
    assert grid.warmup_syntheses == 3


def test_adapter_spec_must_be_importable_shaped(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "model.yaml",
        """
name: broken
adapter: rtb.generate.dummy.DummyAdapter
max_ref_seconds: 6.0
default_sample_rate: 24000
manifest_path: data/manifests/broken.jsonl
""",
    )
    with pytest.raises(ConfigError, match="module.path:ClassName"):
        load_model_config(path)


def test_model_name_with_separator_is_rejected(tmp_path: Path) -> None:
    # El nombre del modelo es un token del generation_id.
    path = write_yaml(
        tmp_path / "model.yaml",
        """
name: xtts__v2
adapter: rtb.generate.dummy:DummyAdapter
max_ref_seconds: 6.0
default_sample_rate: 24000
manifest_path: data/manifests/x.jsonl
""",
    )
    with pytest.raises(ConfigError):
        load_model_config(path)


def test_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_phrases(tmp_path / "absent.yaml")
