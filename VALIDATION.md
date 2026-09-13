# Validation de la version 0.2.3

- Interruption utilisateur simulée avec le comportement natif Wan2GP (`generate_media` retourne `True` tandis que `state.gen.abort` reste actif) : aucune erreur et capture temporaire nettoyée.
- Fin normale simulée sans checkpoint : erreur de sécurité toujours émise.

- Identité publique, famille affichée, quatre noms de modèles, dossier racine et archive renommés en **H3 Latent Continue** / `wan2gp-h3-latent-continue`.
- Identifiants d'architecture et clé des options inchangés pour maintenir la compatibilité des données existantes.

- 31 tests CPU réussis ; les deux tests de comparaison des empreintes des sources ont été retirés avec le contrôle correspondant.
- `pipeline_smoke.py` et `ui_smoke.py` réussis chacun sur les archives Wan2GP 12.72 et 13.0 fournies.
- Le test 13.0 utilise la vraie classe `SharedState` et le helper `service_for`, extraits du code natif. Le service Deepy complet n'est pas démarré.
- Le pipeline utilise le décorateur natif de progression sur 13.0, et l'absence optionnelle de ce module sur 12.72 est testée.
- Le parcours Gradio vérifie qu'une case cochée atteint une tâche figée, puis aboutit à un safetensors dans un dossier personnalisé avec le nom final et les tenseurs exacts.
- Les poids H3, CUDA/MMGP, le codec et le mux audio réels ne sont pas testés localement. Les tests de pipeline utilisent de petits remplaçants CPU.

Depuis la racine du plugin, avec les dépendances de test disponibles :

```bash
python -m unittest discover -s tests -q
python tests/pipeline_smoke.py /chemin/vers/Wan2GP
python tests/ui_smoke.py /chemin/vers/Wan2GP
```

Le diagnostic `check_compatibility.py` est optionnel et ne valide aucune version. Aucune comparaison des sources ne conditionne le chargement.

## Historique : validation 0.2.0

- 33 tests CPU : checkpoints v1/v2, géométrie des contextes, origine audio fractionnaire, isolation des options, export par blocs et nom final.
- `tests/pipeline_smoke.py` exécute le corps du pipeline avec DiT/VAE simulés : modes Reference, Context et Audio Prefix, préfixe audio bit à bit identique après débruitage, durée attendue, rechargement puis nouvelle continuation.
- `tests/ui_smoke.py` exécute les événements Gradio et les corps natifs de sauvegarde du formulaire, mise en file et export vidéo ; les six contrôles du panneau, y compris les paramètres expérimentaux, arrivent dans l'instantané de tâche.
- Les positions audio cibles sont décalées, pas celles des références. Les deux canaux du préfixe reçoivent le timestep de conditionnement fixe.
- Aucun test avec les poids H3 réels ni d'évaluation perceptuelle des nouveaux raccords n'a été effectué ici. Le retour GPU positif de l'utilisateur concerne la 0.1.7.
- `SOURCE_CHANGES.diff` est le diff historique de l'adaptation initiale ; pour les évolutions 0.2.0, les fichiers `seams.py`, `latent_runtime.py`, `pipeline.py`, `layout.py` et le changelog font référence.

## Historique : validation 0.1.1

Environnement des tests : Python 3.12, PyTorch 2.14.0+cpu, safetensors 0.8.0 ; Gradio 5.29.0 pour la construction d'interface. Les dépendances utilisées pour les tests ne sont pas embarquées dans le ZIP et n'imposent pas une mise à jour de l'environnement Wan2GP de l'utilisateur.

## Régression de démarrage corrigée en 0.1.1

- Exécution de `discover_plugin_model_extensions` du gestionnaire natif sur le véritable manifeste : une extension est détectée, son handler est résolu et les dossiers defaults/profiles existent.
- Appel des véritables callbacks du plugin avec des objets Gradio 5.29.0 `State` : restauration, état vide, imbrication, visibilité, modification et préparation de tâche.
- Vérification de la conservation des données d'un autre plugin et de l'absence de mutation de l'état d'origine.
- Suite CPU portée à 20 tests, tous réussis. Les tests spécifiques à la régression ont été exécutés sur la version 0.1.1. Le test avec petits remplaçants du pipeline effectué en 0.1.0 concerne une logique d'inférence inchangée par ce correctif.

## Contrôles effectués

- **20 tests CPU réussis** (`tests/test_core.py`).
- Conservation bit à bit des tenseurs FP32, BF16 et FP16 après sauvegarde/rechargement.
- Refus des LoRA, des mauvaises dimensions et des valeurs non finies.
- Refus d'écrasement d'un checkpoint existant.
- Liaison entre vidéo et checkpoint vérifiée par SHA-256.
- Positionnement des tokens sur la grille H3 et conservation de leur phase à la frontière.
- Alignement des blocs audio et prise en charge des positions négatives/fractionnaires par le placement adapté.
- Isolation des options par tâche et nettoyage du contexte lors d'une exception.
- Conservation des options à travers le nettoyage natif du formulaire, avant la mise en file.
- Nom de checkpoint dérivé du nom final ; empreinte calculée après les métadonnées vidéo.
- Reconnaissance d'une petite coupe automatique en fin de sortie, refus d'une fin modifiée et d'une fin répétitive ambiguë.
- Refus de configurations incompatibles, dont le rendu en deux phases.
- **Test de construction d'interface réussi** avec Gradio 5.29.0 et le code d'insertion natif : deux cases décochées, champ safetensors, position sous la vidéo, aucun onglet principal ajouté, quatre adaptateurs de cycle de vie installés. Les initialisations de packages et la migration de configuration qui importent des moteurs sans rapport sont isolées dans le bootstrap de ce test.
- **Test du corps réel de `generate()` réussi avec petits remplaçants CPU** : première génération, continuation pixels, continuation latente vidéo/audio et capture. La continuation latente ne rappelle ni l'encodeur vidéo ni l'encodeur audio. Ce test utilise des sorties artificielles, pas les poids H3.
- Compilation syntaxique des fichiers Python et vérification des empreintes du snapshot fourni.

## Ce qui n'a pas été validé

- Chargement des véritables poids 20B/33B et des VAE dans le runtime complet.
- CUDA, MMGP/offload, VRAM/RAM de production et combinaison de chaque LoRA/cache/sampler.
- Parcours navigateur dans une session complète Wan2GP, redémarrage et file d'attente réelle.
- Gain de netteté/contraste, absence de dérive, qualité du raccord, synchronisation audio réelle et longues suites.

**Aucun résultat visuel H3, taux de réduction de dégradation ou benchmark GPU n'est revendiqué.** La version est un prototype à tester sur le snapshot indiqué. Le protocole du README sert à établir son intérêt réel avant de poursuivre son développement.
