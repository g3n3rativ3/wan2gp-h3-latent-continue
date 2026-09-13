# H3 Latent Continue — Wan2GP

Version **0.2.2**, continuation latente H3, rendu **en une seule phase exclusivement**.

## Version 0.2.2 : nouveau nom

Le plugin s'appelle désormais **H3 Latent Continue** dans Wan2GP. Son dossier d'installation et le dossier racine du ZIP s'appellent `wan2gp-h3-latent-continue`. Les quatre modèles apparaissent sous le suffixe **Latent Continue**. Les identifiants techniques `h3_latent_*` et la clé interne historique sont conservés afin de maintenir la lecture des checkpoints, presets et tâches déjà créés.

Avant l'installation, supprimer l'ancien dossier `plugins/wan2gp-h3-latent-prototype` pour éviter que Wan2GP charge les deux copies.

## Version 0.2.1 : mises à jour Wan2GP sans blocage par empreinte

Le contrôle des empreintes des sources est supprimé, ainsi que sa liste de fichiers autorisés. Aucun numéro de version, commit ou contenu de fichier Python n'est utilisé pour autoriser un rendu. Une modification de `wgp.py`, l'ajout d'un modèle ou un changement du code MiniMax H3 ne déclenche donc plus de refus par principe.

Les changements de progression du pipeline H3 de l'archive Wan2GP 13.0 fournie ont été intégrés. Sur l'ancienne archive 12.72, l'affichage de progression supplémentaire est simplement absent. Le VAE, le transformer et le chargement des poids restent ceux de l'installation Wan2GP ; le correctif natif de décodage des clips courts est ainsi utilisé sur 13.0.

**Installation :** fermer Wan2GP, supprimer l'ancien dossier `plugins/wan2gp-h3-latent-prototype`, installer `plugins/wan2gp-h3-latent-continue` depuis ce ZIP, redémarrer, puis Ctrl+F5. Ne pas conserver deux copies du plugin. Aucun fichier du cœur Wan2GP n'est à remplacer. Les couples vidéo/checkpoint v1 et v2 existants restent lisibles ; les options et modes expérimentaux de la 0.2.0 sont conservés.

**Contrôles conservés :** association entre vidéo et checkpoint, format et géométrie des tenseurs, modèle/VAE déclarés, cadence et configurations réellement non prises en charge, dont les deux phases. Ces vérifications protègent les données de continuation ; elles ne comparent pas le code source. Un checkpoint incompatible ne provoque jamais de repli silencieux vers le réencodage vidéo.

**Limite :** aucune garantie universelle pour les futures versions. Une fonction supprimée ou une véritable modification des conventions des latents peut demander une adaptation. Le plugin conserve un pipeline adapté : les futures modifications du `generate()` natif ne sont pas automatiquement fusionnées. Pour améliorer cette maintenabilité, l'étape suivante serait de proposer des points d'extension natifs de capture/injection à Wan2GP.

Validation CPU réussie sur les deux archives fournies (12.72 et 13.0), dont événements Gradio, état partagé 13.0, mise en file, écriture du checkpoint et continuation avec les trois modes. Pas de validation locale avec les poids H3 réels, CUDA ou l'ensemble de vos autres plugins. Détails et commandes dans `VALIDATION.md`.

Les sections de correctifs ci-dessous constituent l'historique. Pour l'installation actuelle et la compatibilité, cette section et la section 1 font référence.

## Version 0.2.0 : expériences sur les raccords vidéo/audio

La démonstration utilisateur après cinq continuations a confirmé un gain visuel net avec la 0.1.7 sur la scène testée. Cette version vise le léger raccord visuel restant et surtout la rupture sonore. La conservation des anciens segments sans recompression est laissée de côté, conformément au retour utilisateur. Aucune garantie de raccord invisible ou de continuité musicale n'est donnée avant vos essais GPU.

### Installation et essai rapide

Fermer Wan2GP, remplacer le dossier `plugins/wan2gp-h3-latent-prototype` par celui du ZIP 0.2.0, redémarrer et faire Ctrl+F5. Ne pas installer deux copies simultanées. Utiliser votre vidéo et son checkpoint déjà créés en 0.1.7 : ils sont acceptés. Sélectionner le même modèle prototype, activer Continue Video et **Enable Continue with latent**, puis charger le checkpoint dans le panneau existant.

Le panneau **Continue with latent** contient maintenant **Join experiment**, **Video latent context (frames)** et **Audio latent context (seconds)**. La sauvegarde des latents et l'activation de la continuation restent décochées par défaut. Le mode de raccord reste sur **Reference (0.1.7)** par défaut. Pour une nouvelle vidéo initiale, utiliser Reference ; les deux expériences exigent une continuation latente.

| Mode | Effet |
|---|---|
| Reference (0.1.7) | Chemin de conditionnement antérieur ; les deux nouveaux réglages de durée sont ignorés. |
| Longer video/audio context | Contexte vidéo de 18, 35 ou 52 images, indépendant du chevauchement natif ; contexte sonore de 0,5, 1 ou 2 secondes, placé sur la chronologie de la continuation. |
| Longer context + frozen audio prefix | Même contexte vidéo étendu ; le passé sonore est copié dans un préfixe figé des latents audio cibles, décodé avec la suite, puis retiré à la précision de l'échantillon audio. Exige **No Skipping**. |

Commencer par **Longer context + frozen audio prefix**, **35 images**, **1 seconde**, avec **No Skipping**. Cocher aussi la sauvegarde pour poursuivre la chaîne. Garder résolution, FPS, modèle, LoRA, prompt, seed, nombre de pas et chevauchement natif identiques entre les essais comparés. Un contexte long coûte plus de mémoire/calcul et peut rendre les changements de scène moins faciles ; 52 images n'est pas automatiquement meilleur que 35. La durée effectivement disponible est limitée par le segment latent sauvegardé, pas par la durée totale du MP4.

### Protocole comparatif

À partir du même couple vidéo/checkpoint, produire trois continuations indépendantes : Reference, Longer context, puis Longer context + frozen audio prefix. Mettre **No Skipping pour les trois** afin d'isoler le changement de méthode. Écouter plusieurs secondes avant et après le raccord : rythme, timbre, niveau, clic, silence, reprise d'une syllabe. Regarder mouvement, cadrage, contraste et texture. Ensuite seulement, essayer 18/35/52 images et 0,5/1/2 secondes, un réglage à la fois. Comparer enfin des chaînes de cinq continuations.

Votre configuration précédente combinait une LoRA Turbo et Spectrum. Pour isoler le nouveau mode, conserver d'abord la même LoRA et retirer Spectrum sur les trois branches. Si la continuité sonore reste insuffisante, un essai distinct sans Turbo, avec le réglage d'inférence natif adapté, permettra d'étudier son influence. Ce n'est pas une obligation pour utiliser le plugin.

### Fonctionnement et limites

Le contexte vidéo reste constitué des latents d'inférence originaux placés à leurs positions temporelles : aucun réencodage VAE ni fondu visuel. Cette version n'implémente pas de préfixe **vidéo** figé ; elle teste d'abord davantage d'historique visuel. Le préfixe **audio**, lui, utilise les lignes cibles natives prévues pour un conditionnement fixe. Elles restent inchangées pendant le débruitage. Leur position est décalée vers le passé, tandis que les références éventuelles gardent leur emplacement. Le décodage audio voit le passé et le futur ensemble, ce qui vise aussi les artefacts de bord du VAE. Aucun fondu sonore ni correction artificielle de volume n'est ajouté.

La cible vidéo commence à la dernière image source, que Wan2GP utilise dans son chevauchement. Le son est positionné relativement à cette origine, avec des indices latents entiers à 40 Hz et une origine temporelle éventuellement fractionnaire. Le pré-roll est supprimé à 32 kHz sans ajouter de silence. Les modes étendus arrondissent la longueur latente audio vers le haut avant de recouper à la durée demandée. Cette précision arithmétique ne garantit pas que le modèle poursuivra exactement la même musique.

Le mode audio-prefix refuse Spectrum et les autres caches de saut d'étapes ainsi que les guides audio additionnels. Les autres restrictions demeurent : une seule phase, une fenêtre générée par tâche, pas de changement résolution/FPS, pas de raffinement audio supplémentaire ni de posttraitement invalidant la correspondance. Les références Ref2VA restent distinctes du préfixe ; leur comportement perceptuel reste à tester. Le journal `[H3 Join]` indique le mode, les durées demandées, le nombre de tokens audio figés et la coupe audio calculée.

### Checkpoints et retour à une ancienne version

La 0.2.0 lit les checkpoints **v1** existants et écrit des checkpoints **v2**, avec `audio_origin_seconds`. En mode préfixe audio, le tenseur sauvegardé contient également le passé figé ; l'origine indique où il se situe par rapport au segment vidéo. Cela permet à la continuation suivante de sélectionner les bons latents, sans les réencoder ni perdre le décalage temporel. La taille du préfixe est bornée par le contexte demandé et n'accumule pas toute la chaîne. Les fichiers restent à côté du MP4 final avec le même nom.

La 0.1.7 refuse les nouveaux checkpoints v2. Pour revenir à cette version, conserver vos couples vidéo/checkpoint v1 d'origine. Dans la 0.2.0, le mode Reference peut continuer un checkpoint v2 car il interprète son origine audio. Les réglages de raccord sont mémorisés avec chaque tâche et dans le manifeste de sortie.

### Sources et validation

La réflexion s'appuie notamment sur [H3 Motion Context, sections audio et réglages](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context#why-the-audio-needed-work) : distinction entre référence sonore et continuité temporelle, fenêtre audio indépendante, effets possibles de Turbo/Spectrum. Son code n'est pas incorporé ; cette implémentation utilise les mécanismes du pipeline Wan2GP fourni. Les observations de son auteur ne sont pas des résultats mesurés pour notre plugin.

Tests CPU : conservation exacte du préfixe pendant le débruitage, durée vidéo/audio, origine fractionnaire, rechargement et deuxième continuation, positions des lignes audio cibles en FL2VA/Ref2VA, paramètres reçus par Gradio/file, sauvegarde finale. Les poids et VAE sont simulés dans le test de pipeline : l'amélioration réelle des raccords reste à évaluer sur votre GPU.

## Correctif 0.1.7 : export des listes de blocs vidéo

La 0.1.6 a permis au rendu utilisateur de recevoir `save=True`. Elle révélait ensuite un défaut indépendant : le contrôle d'export attendait un tenseur unique, alors que `wgp.py` assemble `output_video_frames` sous forme de liste de blocs temporels. L'erreur `'list' object has no attribute 'ndim'` se produisait avant l'encodage.

Le contrôle accepte désormais les listes/tuples de blocs RGB uint8 utilisés par l'export SDR de Wan2GP, en C,T,H,W ou T,H,W,C suivant les mêmes règles de disposition que l'exporteur natif. Il compte les images de tous les blocs et extrait seulement les huit dernières images, même lorsqu'elles traversent une frontière entre blocs. La liste originale reste inchangée et est transmise à l'exporteur. Aucune concaténation de la vidéo complète n'est effectuée pour ce contrôle.

Les listes sans aucune image, les changements de résolution, les dispositions non prises en charge et les listes de blocs non uint8 sont refusés explicitement (Wan2GP convertit normalement les sorties SDR en uint8 avant cette étape). Le fichier latent reste lié au chemin final après assemblage audio et métadonnées, et non au fichier vidéo temporaire `_tmp`.

**Installation :** fermer Wan2GP, remplacer le dossier du plugin par celui du ZIP 0.1.7, redémarrer et faire Ctrl+F5. Créer une nouvelle tâche avec la sauvegarde cochée. Les correctifs précédents sont inclus ; les restrictions, notamment le rendu en une seule phase, restent inchangées.

**Validation :** 29 tests CPU ; listes à un ou plusieurs blocs, frontière dans les huit dernières images, bloc source précédant la génération, sortie tronquée, nom final après assemblage audio simulé, empreinte et tenseurs exacts. Le test Gradio exécute également le corps natif de `save_video` pour une liste, avec un writer imageio simulé qui vérifie les images reçues. L'inférence GPU, le codec réel et le mux audio réel restent à valider chez l'utilisateur.

## Correctif 0.1.6 : transport indépendant dans la tâche

Le journal de la 0.1.5 prouve que la case était lue (`Form captured: save=True`) mais que le rendu recevait `save=False`. Le point précis de perte dans l'installation utilisateur n'est pas déterminé : les autres plugins installés ne sont pas inclus dans le snapshot étudié. La correction ne suppose pas qu'un plugin particulier est responsable.

Les options sont désormais copiées dans un enregistrement JSON propre aux paramètres de la tâche (`_h3_latent_task_options_v1`), lors de l'enregistrement du formulaire puis de l'appel natif `add_video_task`. Si le champ partagé `plugin_data` disparaît entre ces étapes, la mise en file retrouve l'instantané capturé pour le modèle. Au rendu, l'enregistrement de la tâche est prioritaire, même si `plugin_data` a été vidé ou si l'utilisateur a décoché la case après la mise en file. L'édition native d'une tâche met également à jour cet enregistrement. Les champs des autres plugins ne sont pas supprimés.

Un appel transitant par une fonction à paramètres `**kwargs` est également correctement interprété. Une tâche de prototype sans enregistrement et sans options explicites dans `plugin_data` est refusée avant le chargement des poids, au lieu de supposer silencieusement que la sauvegarde est désactivée. Pour les anciennes tâches incomplètes, recréer la tâche depuis le formulaire. Les tâches explicitement désactivées restent des comparaisons sans sauvegarde.

**Installation :** fermer Wan2GP, remplacer le dossier du plugin par celui du ZIP 0.1.6, redémarrer et recharger avec Ctrl+F5. Cocher la sauvegarde puis créer une nouvelle tâche. Contrôler les trois étapes : `Form captured: save=True`, `Queued model=...: save=True`, puis `Job options: save=True`. À la fin, `Saved ...safetensors` donne le chemin du fichier. Le nom et le dossier dérivent toujours du MP4 réellement exporté.

**Validation :** 27 tests CPU et test Gradio étendu, avec suppression volontaire de `plugin_data` avant la mise en file ET avant le rendu. Le test obtient un checkpoint dans un dossier personnalisé, vérifie les tenseurs exacts et l'empreinte de la vidéo finale, même après désactivation ultérieure de la case. Les corps natifs d'enregistrement et de mise en file sont exécutés ; l'inférence et l'encodage vidéo restent simulés. Pas de validation GPU avec l'ensemble des plugins de l'utilisateur. Le rendu en deux phases reste exclu.

## Correctif 0.1.5 : lecture des cases dans l'événement natif

Le journal utilisateur de la 0.1.4 montrait `Job options: save=False` malgré la demande de sauvegarde. Les tests précédents validaient les callbacks séparément ; ils ne prouvaient pas que Generate lisait les contrôles affichés. La 0.1.5 ajoute les trois contrôles du panneau aux entrées des événements `save_inputs` déjà enregistrés pour ce formulaire. Leurs valeurs sont lues dans le même événement que les paramètres vidéo, avant l'enregistrement des paramètres et la mise en file. Aucun paramètre du formulaire natif n'est recopié dans un second formulaire.

Les mises à jour asynchrones de `plugin_data` ne constituent donc plus la seule voie de transmission. Un rafraîchissement retournant un `plugin_data` vide n'efface plus les cases et le fichier affichés. Les valeurs d'une tâche déjà mise en file restent un instantané : changer les cases ensuite ne modifie pas cette tâche.

**Installation :** fermer Wan2GP, remplacer le dossier du plugin par celui du ZIP 0.1.5, redémarrer et faire Ctrl+F5. Cocher la sauvegarde et créer une nouvelle tâche. Une ancienne tâche contenant `save=False` reste désactivée.

Messages à vérifier :

- Au lancement : `Inline controls connected to N native form event(s)` (N dépend du formulaire).
- Au clic sur Generate ou à l'enregistrement du formulaire : `Form captured: save=True, continue=False` pour une création avec sauvegarde.
- Au début de la tâche : `Job options: save=True, continue=False`.
- À la fin : `Saved ...safetensors`, avec le chemin final à côté du MP4.

Si aucune liaison d'événement natif n'est trouvée, une erreur explicite est émise au montage du panneau. Cette intégration utilise les événements Gradio déjà construits : elle est testée sur Gradio 5.29.0 et le snapshot Wan2GP fourni. Une future modification du montage des événements peut demander une adaptation. En cas d'échec, joindre le journal complet incluant les lignes `Inline controls`, `Form captured` et `Job options`.

**Validation renforcée :** le test `tests/ui_smoke.py` appelle réellement `Blocks.process_api` avec la case cochée et des données mémorisées fausses, puis les corps natifs `save_inputs` et `add_video_task`. Il poursuit jusqu'à la capture et l'écriture atomique du checkpoint, vérifie les tenseurs exacts et l'empreinte de la vidéo après ajout des métadonnées, dans un dossier de sortie personnalisé. L'inférence et l'encodeur vidéo de ce test sont simulés sur CPU. Ce n'est pas une validation GPU ni une preuve de gain de qualité. Les restrictions de rendu, notamment une seule phase, restent inchangées.

## Correctif 0.1.4 : transmission des options à la file

Les changements explicites des cases et du fichier latent sont désormais mémorisés dans la session, séparément pour chaque modèle. Ils sont fusionnés dans les paramètres lorsque Wan2GP prépare le formulaire et lorsque Generate récupère les paramètres du modèle. Cela évite de dépendre d'une modification d'un autre champ natif après avoir coché la case. Les données des autres plugins sont conservées et les sessions ne partagent pas leurs choix.

Après remplacement du dossier du plugin et redémarrage, recharger la page avec Ctrl+F5. Cocher **Save latent checkpoint next to output video** avant de créer une nouvelle tâche (les anciennes tâches ne sont pas réécrites). La case reste décochée par défaut. Le journal doit afficher `[H3 Latent] Job options: save=True, continue=False` pour une création avec sauvegarde, puis `[H3 Latent] Saved ...safetensors` après l'export. Si `save=False` apparaît, la tâche n'a pas demandé de sauvegarde.

Le chemin du checkpoint est dérivé du chemin final réel du MP4 : seul le suffixe est remplacé par `.safetensors`. Il respecte ainsi le répertoire de sortie utilisé par Wan2GP. Une vidéo déjà produite sans capture ne permet pas de récupérer rétroactivement les latents originaux.

Tests : callbacks Gradio, case modifiée sans autre édition du formulaire, ancien état du formulaire, désactivation et isolation entre sessions. La génération réelle GPU reste à valider. Les restrictions précédentes, dont une seule phase, restent en vigueur.

## Correctif 0.1.3 : prétraitement des LoRA

Le chargement des LoRA transmettait les identifiants `h3_latent_*` au convertisseur AdaLN natif, qui ne connaît que les identifiants `minimax_h3_*`. Le plugin traduit désormais ces identifiants pour les quatre variantes FL2VA/REF2VA, complètes/pruned, sur la seule instance du transformer du prototype. La conversion native des tenseurs reste inchangée et ses erreurs restent propagées. Les identifiants natifs sont transmis tels quels. Aucune modification des fichiers Wan2GP ni des autres instances de modèles.

Fermer Wan2GP, remplacer le dossier `plugins/wan2gp-h3-latent-prototype` par celui du ZIP 0.1.3, puis redémarrer. Les correctifs 0.1.1 et 0.1.2 sont inclus. Le rendu reste exclusivement en une phase. Tests CPU de traduction, isolation et installation répétée ; la génération GPU et la qualité des LoRA réels restent à valider.

## Correctif 0.1.2 : contrôle des sources sous Windows

**Historique uniquement : ce contrôle a été supprimé en 0.2.1. Les consignes de refus ci-dessous ne s'appliquent plus.**

Le contrôle de compatibilité normalise maintenant les fins de ligne CRLF/CR en LF avant de calculer les empreintes des sources Python. Une conversion des fins de ligne par Git sous Windows ne provoque donc plus un faux refus. Les autres différences et fichiers absents restent refusés, avec les empreintes attendues et constatées dans le message. Le script `check_compatibility.py` applique exactement le même contrôle. Les empreintes des vidéos et checkpoints restent calculées sur les octets exacts, sans normalisation.

Ce correctif ne garantit pas que votre installation utilise le snapshot attendu : si le refus persiste, joindre les fichiers Wan2GP signalés et le nouveau message pour permettre une adaptation. Réinstaller le plugin ne remplace pas ces sources Wan2GP. Ne pas supprimer le contrôle.

Installation : fermer Wan2GP, remplacer le dossier `plugins/wan2gp-h3-latent-prototype` par celui du ZIP 0.1.2, puis redémarrer. Aucun changement du pipeline d'inférence ni du format des latents. Le rendu en deux phases reste exclu.

Validation : tests des fins de ligne LF/CRLF/CR, refus d'un changement de code et d'un fichier absent. Pas de validation avec les poids H3 sur GPU.

Ce plugin de modèle ajoute une continuation depuis les latents vidéo/audio d'origine, dans le formulaire **Media Generator** existant de Wan2GP. Il ne crée pas de nouvel onglet principal et ne recopie pas les paramètres du générateur.

**Statut : tests CPU et construction de l'interface réalisés ; aucune génération avec les véritables poids H3 sur GPU n'a été exécutée lors de la préparation de cette version. L'amélioration visuelle reste à mesurer.** Ce prototype ne doit pas être considéré comme une solution démontrée à toute dégradation de l'image.

## Correctif 0.1.1 et remplacement de la version 0.1.0

Cette version corrige deux défauts de la livraison initiale :

- `plugin_info.json` déclare maintenant `profiles: "./profiles"`, et le dossier correspondant est inclus. Le chargeur réel du snapshot exige les deux chemins `defaults` **et** `profiles` ; leur déclaration incomplète empêchait la découverte des modèles malgré le chargement de la partie interface.
- Les données des callbacks peuvent être des dictionnaires ou des objets Gradio `State` lors d'un rafraîchissement. Leur valeur est maintenant extraite avant la lecture/restauration, la modification des cases et la préparation de la tâche. Les données des autres plugins sont conservées. Les états vides ou mal formés donnent les options désactivées par défaut.

**Mise à jour sous Windows :** fermer Wan2GP, extraire le nouveau ZIP dans `K:\Wan2GP\plugins\` et remplacer les fichiers du dossier existant `wan2gp-h3-latent-prototype`. Ne pas installer une seconde copie du plugin dans un autre dossier. Redémarrer Wan2GP puis recharger complètement la page du navigateur (Ctrl+F5). Les vidéos et checkpoints de sortie ne sont pas concernés par ce remplacement. Le format de checkpoint reste en version 1.

La découverte du modèle et les callbacks recevant de vrais `gr.State(...)` sont désormais testés, en plus de la construction des composants. Ce correctif ne revendique pas une validation GPU ni une compatibilité vérifiée avec chaque combinaison de plugins tiers.

## 1. Compatibilité avec Wan2GP

Source : archive fournie `Wan2GP-main(1).zip`, commentaire Git de l'archive :
`362c3467a70e1136ceb52eec95907205a8f88543` (snapshot daté du 7 septembre 2026).

La version 0.2.1 est également testée sur l'archive fournie `Wan2GP-main(2).zip` (13.0). Ces archives sont des bases de validation, pas une liste de versions autorisées. Les sources originales de Wan2GP ne sont ni écrites ni remplacées.

Aucun contrôle de version ou d'empreinte du code n'est exécuté au chargement. Les appels utilisent les signatures des fonctions installées lorsque les adaptateurs les délèguent ; les erreurs réelles de ces fonctions restent propagées.

Diagnostic optionnel de présence de l'installation, sans charger les modèles :

```bash
python plugins/wan2gp-h3-latent-continue/check_compatibility.py .
```

À exécuter depuis la racine de Wan2GP avec son interpréteur Python. Ce script vérifie uniquement la présence de trois fichiers d'entrée, ne certifie pas la compatibilité et n'intervient jamais dans le rendu.

## 2. Installation

1. Fermer Wan2GP.
2. Extraire le ZIP. Il contient un unique dossier racine : `wan2gp-h3-latent-continue`.
3. Placer ce dossier directement dans `Wan2GP/plugins/` : le fichier doit se trouver à `Wan2GP/plugins/wan2gp-h3-latent-continue/plugin.py`.
4. Redémarrer Wan2GP, ouvrir **Plugins**, activer **H3 Latent Continue**, enregistrer et redémarrer à nouveau.
5. Dans le sélecteur de modèles habituel, choisir une entrée **MiniMax H3 … — Latent Continue**.

Modèles proposés : FL2VA 33B, Ref2VA 33B, FL2VA Pruned 20B, Ref2VA Pruned 20B. Les URL de poids viennent du snapshot fourni. Les poids ne sont pas dans le ZIP ; le chargeur normal de Wan2GP les résout/télécharge. Les LoRA et profils H3 habituels sont réutilisés via le gestionnaire natif. Les paramètres spécifiques aux configurations non prises en charge restent soumis aux restrictions ci-dessous.

Les entrées H3 ordinaires restent disponibles. Le prototype utilise des identifiants d'architecture distincts, sans remplacer ceux des modèles natifs.

Dépendances : environnement Wan2GP existant, notamment PyTorch, safetensors et Gradio **5.29.0** dans le snapshot. Ce plugin n'installe pas une autre version de PyTorch ou Gradio. Aucun ComfyUI ni nœud ComfyUI n'est requis.

## 3. Interface

Sous le champ vidéo habituel se trouvent les contrôles propres au modèle prototype :

- **Save latent checkpoint next to output video** : sauvegarder un fichier latent associé. Décoché par défaut.
- Un panneau repliable **Continue with latent**, visible lorsque **Continue Video** est sélectionné.
- Dans ce panneau : **Enable Continue with latent (experimental)**, décoché par défaut, et un champ d'import **Matching H3 latent checkpoint (.safetensors)**.

Le champ **Video to Continue** reste le champ d'origine. Les autres réglages — prompt, résolution, seed, LoRA, sampler, nombre de pas, références, etc. — restent dans le formulaire existant. Il faut utiliser les entrées de modèle portant « Latent Continue » pour voir ces contrôles.

Les options sont attachées aux données de la tâche lors de la mise en file. Une tâche déjà ajoutée ne doit pas dépendre des modifications ultérieures des cases. L'édition/restauration des données du formulaire synchronise les contrôles ; les fichiers importés restent soumis à l'existence de leur chemin local. Après nettoyage des fichiers temporaires de Gradio ou déplacement vers une autre machine, réimporter les fichiers. Ne pas considérer une archive de file d'attente comme une sauvegarde autonome de tous les médias.

## 4. Produire une vidéo et son fichier latent

1. Choisir le modèle prototype correspondant au H3 souhaité.
2. Utiliser une phase. L'option deux phases n'est pas prise en charge, y compris dans le périmètre envisagé d'une éventuelle version finale.
3. Cocher **Save latent checkpoint next to output video**.
4. Laisser **Enable Continue with latent** décoché pour la première vidéo.
5. Générer normalement, avec les post-traitements incompatibles désactivés.

Pour une sortie `output/ma_video.mp4`, le plugin écrit :

- `output/ma_video.mp4` : la vidéo finale normale ;
- `output/ma_video.safetensors` : le checkpoint latent associé.

Le dossier est le véritable dossier de sortie configuré dans Wan2GP, même s'il ne s'appelle pas `output`. Le nom est dérivé du nom final choisi par Wan2GP, après résolution des conflits de noms. Le fichier latent n'est pas un LoRA et n'est pas ajouté au sélecteur de LoRA.

Le checkpoint n'est publié qu'après l'enregistrement de la vidéo et de ses métadonnées. L'écriture utilise un fichier temporaire puis un remplacement atomique ; un checkpoint existant n'est jamais écrasé. Une erreur de sauvegarde est signalée : une vidéo qui existe sans son `.safetensors` n'est pas annoncée comme une sauvegarde latente réussie. Consulter la console si la vidéo existe mais que le fichier associé est absent.

Message attendu : `[H3 Latent] Saved ...`.

## 5. Continuer plus tard

1. Reprendre la même variante de modèle prototype, le même checkpoint et le même VAE.
2. Sélectionner **Continue Video** dans le formulaire habituel.
3. Charger la vidéo d'origine dans **Video to Continue**.
4. Ouvrir **Continue with latent**, importer son `.safetensors` puis cocher l'activation.
5. Conserver exactement la résolution et la cadence de la génération précédente.
6. Choisir le nouveau prompt et les paramètres de génération souhaités.
7. Cocher aussi **Save latent checkpoint…** si cette nouvelle sortie doit être continuée ensuite.
8. Générer une seule nouvelle fenêtre.

Le recouvrement habituel reste utilisé. Commencer par **18 images** ; 1 image est admise mais fournit moins de contexte de mouvement. L'overlap ne doit pas dépasser la partie générée finale enregistrée dans le checkpoint.

La console indique que le contexte vidéo est chargé directement, sans réencodage VAE. La voie latente ne se rabat jamais silencieusement sur la voie pixels si un fichier est absent ou incompatible.

**Le fichier sauvegardé décrit les latents de la dernière fenêtre générée, pas un latent global reconstruit de toute la vidéo assemblée.** Il contient les données nécessaires à une reprise à la fin. Le lien avec l'ensemble de la vidéo est enregistré et vérifié. Une reprise à une position arbitraire dans un ancien segment n'est pas prise en charge.

Une ancienne vidéo enregistrée sans checkpoint ne permet pas de retrouver rétroactivement ses vrais latents d'inférence. Réencoder son MP4 produirait une reconstruction, pas les données d'origine. Le plugin ne présente pas cette reconstruction comme un checkpoint authentique.

## 6. Ce qui est sauvegardé et réutilisé

Le format safetensors est identifié par `format=wan2gp.h3.latent-continuation`, version 2 depuis la 0.2.0 (lecture de la version 1 conservée). Il contient :

| Élément | Contenu |
|---|---|
| `video` | Tenseur final du débruitage, forme `[1,24,T,H/16,W/16]`, capturé avant la conversion de dtype destinée au VAE |
| `audio` | Tenseur audio final `[1,32,2,A]` |
| `last_frame` | Dernière image avant compression, pour la présentation au text encoder ; ce n'est pas un latent de conditionnement réencodé |
| Métadonnées | Modèle/VAE déclarés, normalisation, dimensions, FPS, fin réellement livrée, paramètres principaux, LoRA, version du format et empreinte de la vidéo |

Les tenseurs vidéo/audio gardent leur dtype d'origine. Pas de conversion supplémentaire en BF16/FP16 pour réduire le fichier. Les tests vérifient une égalité exacte après sauvegarde/rechargement en FP32, BF16 et FP16.

L'identification des poids repose sur leur variante et leurs noms configurés, pas sur le hachage intégral des dizaines de Go de poids. Ne pas remplacer les poids par d'autres contenus sous le même nom. L'empreinte de la vidéo, elle, est calculée sur le fichier complet après écriture de ses métadonnées.

Les blocs vidéo couvrant la fin livrée sont récupérés directement et positionnés individuellement. La grille temporelle H3 est `(1,4,4,4,4)` images par token, soit 17 images pour 5 tokens. On conserve la phase et la position des tokens d'origine ; on ne traite pas un token final comme s'il représentait une image isolée.

Le contexte audio est également découpé directement dans le latent sauvegardé et positionné par rapport au début de la dernière image source, sur sa grille à 40 Hz. Une adaptation locale du placement audio accepte ces décalages, y compris fractionnaires. La synchronisation perceptuelle doit encore être validée avec H3 réel.

La légère augmentation de bruit de conditionnement du pipeline natif (`0.999`) reste en place. « Latents d'origine » décrit la source sauvegardée, pas l'absence de toute transformation mathématique lors de la préparation du conditionnement.

## 7. Configurations refusées et limites

- **Deux phases : refusées.** Pas de repli automatique ; elles ne sont pas prévues pour une version finale.
- Une seule fenêtre générée par tâche. La génération de plusieurs continuations reste possible en rechargeant chaque nouvelle sortie. Si une tâche demande une deuxième fenêtre, elle est arrêtée avant cette deuxième inférence ; une première sortie peut déjà exister.
- Une seule vidéo par batch. Les répétitions de tâches restent distinctes.
- Pas de PDD, VDN, TTS, Viggle ou modèles externes à ces quatre entrées.
- Pas de changement de résolution/FPS entre source et continuation latente.
- Pas de découpage manuel de la vidéo source, de recadrage, de source remontée ou recompressée. Son empreinte doit correspondre.
- Pas de super-résolution, interpolation, grain ajouté, correction couleur de fenêtre, self-refiner, montage par masque/vidéo de contrôle, traitement audio/remplacement de voix, HDR ou phase supplémentaire de raffinement audio lorsque les fonctions latentes sont actives.
- Les petites coupes de fin automatiques de Wan2GP sont reconnues par comparaison des dernières images avant compression avec le décodage. La correspondance utilise une suite de 8 images. Une fin entièrement répétitive donnant plusieurs positions possibles est refusée. Au-delà des 64 dernières positions examinées, ou si la correspondance échoue, la sauvegarde est refusée. Aucun fichier latent avec un point de reprise deviné n'est écrit.
- Le fichier doit être un checkpoint de ce plugin, de version et de formes reconnues. Un LoRA, un fichier tiers, un tenseur non fini ou un fichier dépassant 2 GiB est refusé.
- Un fichier déplacé/renommé reste utilisable si son contenu vidéo ne change pas et si le bon checkpoint est fourni. Un remuxage ou une modification des métadonnées après sauvegarde change l'empreinte et entraîne un refus.
- L'assemblage reste celui de Wan2GP : les anciennes parties peuvent être réencodées lors d'une continuation. Le plugin évite cette compression dans la source du conditionnement latent, mais ne fournit pas encore un assemblage sans pertes successives des anciens segments.
- Une retouche après sauvegarde n'est pas automatiquement représentée dans les latents. Ne pas remplacer la vidéo liée par une version retouchée.
- Le fichier latent et l'image auxiliaire occupent de la RAM CPU et de l'espace disque. À 1344×768 et 124 images, compter environ 26 Mo pour vidéo latente FP32 + image auxiliaire FP32 + audio, hors petites métadonnées. Le contexte, et non tout le film historique, entre dans la prochaine inférence.

La qualité du raccord dépend aussi du VAE temporel, de ses chevauchements, du modèle et du contexte. Les nouvelles images peuvent encore perdre des détails ou dériver en identité/contraste. Un latent intact ne garantit pas un rendu futur intact.

## 8. Protocole comparatif recommandé

### Préparer une source commune

Générer un clip A avec « Save latent checkpoint » activé. Conserver A et son safetensors. Noter modèle, VAE, résolution, FPS, LoRA, sampler, nombre de pas, flow shift et seed. Désactiver les accélérations par cache pour un premier test interprétable, puis les réintroduire séparément.

### Comparer la première continuation

- B-pixels : reprendre **A**, laisser « Enable Continue with latent » décoché, enregistrer avec un nouveau nom.
- B-latent : reprendre **le même A**, avec son checkpoint et la case activée. Même prompt, seed, modèle, LoRA et paramètres que B-pixels.
- Sauvegarder les latents des deux résultats si l'on souhaite prolonger ensuite chaque branche indépendamment.

Même seed ne signifie pas mêmes images : la représentation et le nombre de blocs de contexte diffèrent, donc la consommation des nombres aléatoires peut aussi différer. Ce comparatif mesure les deux modes de continuation complets ; ce n'est pas une ablation parfaitement isolée du seul VAE.

Examiner le raccord, les premières nouvelles images et la fin du nouveau segment. Comparer détails fins, peau, tissus, cheveux, contraste, couleurs, identité, fluidité et son. Répéter avec plusieurs sources et seeds. Ne pas juger sur une seule scène heureuse.

### Accumulation

Produire par exemple 5 continuations successives dans chaque branche, avec les mêmes prompts/paramètres. Chaque branche doit utiliser son propre résultat précédent. Une source reste inchangée durant une comparaison de candidats.

Pour un deuxième essai, utiliser les références originales dans le formulaire Ref2VA pour les deux branches. Ne pas ajouter une référence à une seule branche, sauf si l'objectif est explicitement de tester cette combinaison.

**Critère de poursuite du projet : amélioration visuelle nette et reproductible dès la première continuation, sans dégradation inacceptable du mouvement/raccord.** Si la qualité reste aussi mauvaise, la sauvegarde latente seule ne remplit pas l'objectif ; diagnostiquer les causes restantes avant d'enrichir l'interface.

## 9. Architecture et maintenance

- `handler.py` : quatre architectures distinctes, délégation au gestionnaire H3 natif et vérification du snapshot.
- `pipeline.py` : copie adaptée du pipeline du ZIP, avec insertion de la préparation latente et de la capture avant décodage. Les autres fonctions restent issues de ce snapshot.
- `packing.py` et `layout.py` : placement des blocs de contexte, notamment des coordonnées audio explicites ; appliqués aux instances prototype.
- `latent_runtime.py` : contexte isolé par tâche, sélection temporelle et capture CPU.
- `checkpoint.py` : format, validation, hachage et sauvegarde.
- `plugin.py` : insertion d'interface sans nouvel onglet principal.
- `integration.py` : quatre adaptateurs installés en mémoire via `set_global`, limités au prototype : conservation des options dans `prepare_inputs_dict`, contexte d'exécution dans `generate_media`, vérification de l'image exportée dans `save_video`, publication après `record_file_metadata`.

Le hook de métadonnées standard ne donne pas accès aux latents et ne suffit pas à associer la capture au nom final. Le pont de cycle de vie est donc nécessaire dans ce snapshot. Ce n'est pas une modification des fichiers du cœur, mais ce pont dépend de fonctions internes ; le plugin n'est pas indépendant des mises à jour de Wan2GP. Un autre plugin qui remplace ces mêmes fonctions doit être vérifié pour compatibilité.

Le nettoyage natif du formulaire retire `plugin_data`. L'adaptateur préserve explicitement une copie des options pour les paramètres de tâches prototype ; sans cette adaptation, les cases seraient affichées mais leurs valeurs n'arriveraient pas au générateur.

Les fichiers safetensors sont écrits à la fin d'une sortie normale. Une interruption ne fournit pas un checkpoint de reprise du débruitage interrompu. Il s'agit d'une reprise de vidéo finie, pas d'une reprise à un pas intermédiaire du sampler.

## 10. Améliorations prioritaires

1. **Essais H3 réels** avec FL2VA puis Ref2VA, mêmes sources, plusieurs seeds ; conserver les résultats bruts et le journal.
2. Comparaison plus isolée avec un mode pixels réencodés disposant exactement de la même géométrie de conditionnement et de la même séquence de bruit que la voie latente.
3. Validation perceptuelle du raccord VAE et de la position des blocs à toutes les fins de clip ; comparaison avec un préfixe latent figé si nécessaire.
4. Conservation des segments maîtres et assemblage différé pour éviter les recompressions du passé.
5. Références originales stables, mesurées séparément, pour limiter la dérive sémantique.
6. Tests de synchronisation audio, limites de contexte et longues suites ; améliorer les contrôles avant ajout en file.
7. Hooks officiels de capture et de fin d'export dans Wan2GP, afin de réduire les adaptateurs internes.
8. Vérification optionnelle des empreintes complètes des poids et meilleur transport des fichiers avec les tâches sauvegardées.

**Le rendu en deux phases est exclu de cette liste et du périmètre futur demandé.**

Pistes comparées, sans dépendance directe à leur code :

- [H3 Motion Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) : contexte et sauvegarde/rechargement de latents.
- [H3 Context Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop) : checkpoints, reprise et assemblage.
- [H3 Inpaint Tools](https://github.com/panghea/ComfyUI-MiniMax-H3-Inpaint-Tools) : extension avec préfixe préservé.

## 11. Tests et limites de validation

Depuis le dossier du plugin, avec Python et les dépendances de Wan2GP :

```bash
python -m unittest discover -s tests -v
python tests/ui_smoke.py /chemin/vers/Wan2GP
python tests/pipeline_smoke.py /chemin/vers/Wan2GP
```

Le test UI construit de vrais composants Gradio 5.29.0 et utilise le corps de l'API d'insertion du snapshot ; son bootstrap isole les imports de moteurs sans rapport avec l'interface. Il ne lance pas l'application complète.

Le test pipeline exécute le véritable corps de `generate()` du prototype avec de minuscules remplaçants CPU du DiT et des VAE. Il vérifie génération, continuation pixels, continuation latente vidéo/audio et capture, ainsi que l'absence d'appel aux encodeurs vidéo/audio dans cette dernière voie. **Ces remplaçants ne produisent pas d'images H3 et ne prouvent pas la qualité.**

Voir `VALIDATION.md` pour les contrôles effectués. Restent à valider : chargement réel des poids, CUDA/MMGP, affichage dans une session complète Wan2GP, file d'attente réelle et surtout l'effet sur la qualité d'image.

## 12. Désactivation et diagnostic

Décocher les deux cases rétablit la voie de conditionnement pixels du pipeline prototype en une phase. Pour retirer tout le plugin, le désactiver dans Plugins puis redémarrer Wan2GP ; ses adaptateurs en mémoire disparaissent. Les fichiers vidéo/latents existants ne sont pas supprimés.

- **Panneau absent** : vérifier l'activation après redémarrage et choisir une entrée « Latent Continue » ; vérifier la console d'insertion UI.
- **Source différente** : utiliser le MP4 original associé ; réimporter les deux fichiers.
- **Dimensions/FPS différents** : reprendre les réglages originaux, sans redimensionner le latent.
- **Modèle/VAE différents** : reprendre la même variante et les mêmes fichiers de poids.
- **Ancien message “source compatibility check failed”** : vous exécutez encore une ancienne version ; remplacer entièrement le dossier du plugin par la 0.2.1 et redémarrer Wan2GP.
- **Output tail no longer matches** : désactiver les traitements/coupes incompatibles ; ne pas forcer la publication d'un fichier associé.
- **Vidéo créée mais latent absent** : consulter l'erreur de sauvegarde ; choisir un nom de sortie neuf si un safetensors existe déjà, puis régénérer avec l'option activée.

Licences et provenance : voir `THIRD_PARTY.md` et les fichiers de licence inclus.
