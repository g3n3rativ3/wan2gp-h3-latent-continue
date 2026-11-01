# 0.6.5

- Wan2GP 19.02: the audio-length and audio-noise edits re-anchored with per-version variants (19.02 names the tail
  `audio_tail_latents` and adds the Pika H3 Flex sampler); 18.19 and 18.17/18.18 variants kept.
- Pika sampler (`pika_renoise`, 6 steps): save, continuation in all join modes and Exact join supported (pinned latents
  and frozen audio exact); "Audio noise as WanGP 18.18" does not change Pika's own noise (Pika has no audio tail).
- Harness: Pika module loaded from WanGP, `ptd` attribute, plugins.py `__file__`, attention_kit SageAttention stubs.

# 0.6.4

- "Audio noise as WanGP 18.18" also applies alone, to every H3 render (Text Only, Start with Image, Continue Video,
  sliding windows): only the seeded audio noise changes; no latent operation, restriction, checkpoint or NaN stop.

# 0.6.3

- Exact join: the new window length is rounded DOWN to an H3 length (5+17n). Starting the window on the VAE chunk
  boundary leaves e.g. 349 frames for a 362-frame WanGP window; native H3 rounded that up to 362, so the model
  composed a longer scene (dialogue included) and the last 13 frames were cut at delivery. Now the scene is generated
  for exactly the delivered frames; the new part is up to 16 frames (4 with a 5+17n source) shorter than requested.
  Requested/generated counts logged and recorded in effective_context.pinned.

# 0.6.2

- NaN/inf produced by the model during a render with latent options is reported explicitly, with the stage (end of
  phase 1, before the audio refinement, end of denoising) and the affected video/audio latents; the render stops
  before decoding instead of failing later with "Ambiguous output tail". Checkpoints were already checked on load.

# 0.6.1

- Option "Audio noise as WanGP 18.18": seeded audio noise drawn without the 18.19 audio tail (channel-major rows as
  before), tail noise from its own generator; same rule for the refinement re-noise and er_sde step noise.
  Off by default; option captured, restored and transported; recorded as settings.audio_noise in the checkpoint.

# 0.6.0

- Exact join in two phases: phase 1 follows the final video (pinned frames re-encoded at phase-1 size with keep_all_latents,
  context from the final latents reduced to phase 1); option, default on; single-phase checkpoints usable in two phases.
- Real-encoder causality check in tests/vae_seam_check.py; harness encoder stand-in returns pixel-dependent latents.

# 0.5.3

- Wording: pinned frames and seamless end follow WanGP's "Windows Frames Overlap"; console states the slider value needed.

# 0.5.2

- Exact join "seamless end" option (default on): the source's last partial VAE chunk is replaced in the complete video
  by the new window's re-decode; export-side consistency checks; second part starts at the replaced frames.
- UI checkbox, form capture, settings restore and task transport for `seamless_end`.
- tests/vae_seam_check.py on the real H3 decoder code; export test with real ffmpeg; UI smoke updated.

# 0.5.1

- Audio guides allowed with the frozen audio prefix and Exact join (Ref2VA voice references); FL2VA soundtrack
  conditioning still refused, with an explicit message.
- Harness: sample-major waveforms (as WanGP passes them), Ref2VA image + voice references in all join modes with the
  packed layout checked including references, FL2VA soundtrack refusal.

# 0.5.0

- New join mode "Exact join" (pinned): window on a VAE chunk boundary, first target latents pinned to the saved
  latents via native masked denoising (phase 1 and phase 2), frozen audio prefix shifted with the window,
  context keyframes before the pinned region; form denoising/masking strengths forced to 1 for the pin.
- Diagnosis of the visible seam documented (no pinned target frame; out-of-distribution context placement).

# 0.4.2

- Wan2GP 18.19: audio-length edit re-anchored with per-version variants (18.19 audio tail / 18.17).
- native_edits.json supports {"variants": [...]} entries (first unique match wins).
- Second part keeps the audio tail past the last frame, like the complete video.
- Harness expects the native audio tail; part2 test with a 100 ms tail.

# 0.4.1

- Two-phase continuation when the phase-1 size changed since the source render: saved phase-1 latents resampled
  (bilinear, antialiased when shrinking) to the current phase-1 size; phase 2 unchanged (exact). Logged and recorded.
- Latent generate compiled at the native line numbers (readable tracebacks).

# 0.4.0

- Allow the Audio Refinement Extra Phase (native and H3 Audio Refinement Controls 1.1.0).
- Latent join kept during refinement: saved-latent audio conditions, frozen prefix restored after re-noise, prefix time offset.
- Compose with H3 Audio Refinement Controls' in-place generate rewrite (exact schedule, kept audio references);
  installed body verified by bytecode; warning for unknown in-place rewrites.
- Harness: real module globals, optional refinement plugin argument, refinement passes recorded; tests/test_refinement.py.

# 0.3.9

- Continue Video: also save the new part alone as `<name>_part2.<container>` (native encoder settings, matching audio segment).
- Start frame located from exact tail fingerprints (handles WGP tail trimming); overlap frames excluded.
- Failure to write the second part is a warning, never a render failure; complete video and checkpoint unchanged.
- New tests/part2_smoke.py with the real native save_video/mux helpers and ffmpeg.

# 0.3.8

- Two-phase H3 support for save and continue (untiled phase 2).
- Checkpoint: optional `video_phase1` tensor and `phase1`/`guide_phases` manifest fields, validated on load.
- Per-stage latent conditions (phase 1 low resolution, phase 2 full resolution); stage-sized prompt frame.
- Manifest width/height are the target size in both modes; settings record `sampling_steps2` and `switch_threshold`.
- Explicit refusal of a two-phase continuation from a single-phase checkpoint.
- Harness: latent-upscaler stand-in; every transformer step builds the native packed layout and checks condition rows.

# 0.3.7

- Fix native H3 hook installation on Wan2GP 18.17 (native `audio_time_offset` / audio join repair).
- Re-anchor the audio-timeline, audio-decode and transformer-layout edits; native offset kept in all formulas.
- Latent continuation resets a non-zero native audio repair offset to 0 (logged); native jobs unchanged.
- Test harness updated for 18.x (Veda generation context, uint8 VAE decode, encoder device argument,
  renamed `settings_error`, workspace binding); new audio-repair regression.

# 0.3.6

- Fix native H3 hook installation after Wan2GP 13.141 added ControlNet cleanup variables.
- Anchor pre-decode capture to the decoding branch, independent of temporary-variable cleanup.
- Preserve capture before VAE dtype conversion; no new source/version allowlist.
- Explicitly reject active ControlNet-Union layers for latent operations.
- Add Ref2VA save/continue and FP16-decoder capture regressions; verify old/new native pipelines.

# 0.3.5

- Support Wan2GP 13.14 strict top-level setting validation: move snapshots into plugin_data and the runtime task envelope.
- Migrate legacy snapshots at native boundaries; preserve disabled passthrough, queued options, imports and task edits without duplicate keyword arguments.
- Retain native unknown-setting rejection and existing local H3 injection checks.
- Extend frozen audio-prefix guard to audio_guide3; reject kept soundtracks only for active latent operations.
- Validate latest native H3 pipeline/packing/layout with CPU stand-ins and native Gradio form/queue/export wiring.

# 0.3.4

- Remove the remaining shared plugin_data.change subscription.
- Restore controls through scalar notifications from existing form-data writers.
- Preserve foreign payloads and callback execution style; no inference changes.
- Add cyclic plugin-data regression and refresh adapter contract tests.

# 0.3.3

- Remove global state.change subscription, which triggered recursive Gradio hashing on native queued/imported state.
- Keep model/mode-driven visibility and all generation behaviour.
- Reproduce the previous RecursionError and test repeated state-return callbacks after native queue creation.

# 0.3.2

- Reference context no longer collapses to one video latent block when native assembly overlap is one frame: at least 18 frames of available latent AV history are retained.
- Context selection and native LoRA/cache settings recorded in the checkpoint; no pixel re-encode added.
- CPU regression covers overlap=1 with unchanged assembly duration. Visual improvement requires a GPU comparison.

# 0.3.1

- Preserve closure-based wrappers without `__wrapped__` by following their captured native function references.
- Regression test: undecorated closure around native H3 generation, all continuation modes and existing loader/layout tests.

# Changelog

## 0.3.0

- Extension-only integration with native MiniMax H3 models; separate model declarations removed.
- Local edits to installed native generation, packing and layout; existing delegating wrappers preserved.
- Original native rendering when options are off; latent operations retain single-phase restrictions and abort handling.
- Recreate old jobs using native model entries. Existing checkpoint formats and native identities remain readable.

# Historique

## 0.2.3

- Un arrêt demandé par l'utilisateur ne déclenche plus l'erreur de checkpoint latent absent.
- La capture latente incomplète est abandonnée et aucun safetensors n'est écrit.
- Le contrôle reste actif lorsqu'un rendu signale une fin normale sans produire le checkpoint demandé.

## 0.2.2

- Nouveau nom public **H3 Latent Continue** dans la liste des plugins et le sélecteur de modèles.
- Nouveau dossier d'installation et dossier racine du ZIP : `wan2gp-h3-latent-continue`.
- Conservation des identifiants de modèles `h3_latent_*`, de la clé interne et des formats de checkpoints pour assurer la continuité avec les rendus précédents.

## 0.2.1

- Suppression complète du blocage par empreinte des sources : aucun fichier Python ni numéro de version n'est comparé avant le rendu.
- Suppression de `compatibility.json` et `source_compatibility.py`. Le diagnostic manuel ne vérifie plus que la présence de l'installation et n'est jamais appelé par le rendu.
- Intégration des changements du pipeline de Wan2GP 13.0 fourni : suivi de progression, filtres de sources Viggle vides et statuts de décodage. Progression optionnelle sur 12.72.
- Tests du formulaire/file/export sur le nouvel état `SharedState`, et maintien des trois modes de continuation et des checkpoints v1/v2.
- Contrôles de données et configurations conservés, notamment une seule phase obligatoire. Aucune garantie de compatibilité avec de futures ruptures d'interface ou de conventions des latents.

## 0.2.0

- Trois modes comparatifs : référence 0.1.7, contexte AV étendu, contexte vidéo étendu avec préfixe audio figé.
- Historique vidéo indépendant du chevauchement natif ; durée audio indépendante.
- Préfixe audio immuable, placement temporel cible, décodage conjoint et coupe à l'échantillon ; caches refusés pour ce mode.
- Checkpoints v2 avec origine audio, lecture des v1 ; réglages transmis dans le formulaire et les tâches.
- Tests de géométrie, préfixe intact, durée de sortie et continuation successive. Validation GPU/perceptuelle à réaliser.

## 0.1.7

- Correction de l'erreur `.ndim` sur la liste de blocs fournie par l'export vidéo natif.
- Lecture de huit images finales au maximum, comptage complet et conservation de la liste originale.
- Tests de blocs multiples, de nom final après mux simulé et du corps natif de l'exporteur ; 29 tests CPU.

## 0.1.6

- Instantané des options propre à chaque tâche, indépendant du champ partagé `plugin_data`.
- Raccord à `add_video_task`, diagnostic de mise en file et prise en charge de l'édition des paramètres de tâche.
- Extraction des options transmises par `**kwargs` ; refus explicite des tâches sans options récupérables.
- Régression testée en supprimant volontairement les données partagées avant la file et avant le rendu ; 27 tests CPU.

## 0.1.5

- Lecture directe des cases et du fichier latent dans les événements natifs `save_inputs` du formulaire concerné.
- Un rafraîchissement contenant des données de plugin vides ne réinitialise plus les contrôles.
- Diagnostic de liaison au démarrage et de capture des valeurs du formulaire.
- Test réel `Gradio.process_api` jusqu'à la file native puis au checkpoint CPU, avec état obsolète et vérification des tenseurs et du chemin final.

## 0.1.4

- Ajout d'un instantané des options par session et modèle et d'un diagnostic des options reçues par la tâche. Insuffisant pour le cas utilisateur, repris en 0.1.5.

## 0.1.3

- Traduction des identifiants des quatre architectures du prototype pour le prétraitement natif des LoRA.

## 0.1.2

- Normalisation des fins de ligne des sources Python dans le contrôle de compatibilité Windows.

## 0.1.1

- Ajout du chemin `profiles` obligatoire pour la découverte des modèles et du dossier correspondant.
- Normalisation des dictionnaires/objets Gradio State dans les callbacks et la préparation des tâches.
- Préservation des données des autres plugins ; valeurs vides ou invalides traitées comme options désactivées.
- Tests de découverte via le gestionnaire natif et tests des callbacks avec de vrais objets State.
- README et instructions de remplacement mis à jour.
- Aucun changement à la logique de débruitage ou au format de checkpoint (version 1).

## 0.1.0

Prototype initial de continuation latente vidéo/audio dans le formulaire de génération existant, avec sauvegarde des latents à côté de la vidéo. Rendu une phase uniquement.
