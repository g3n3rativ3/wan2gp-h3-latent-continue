# Provenance et licences

Ce prototype contient une adaptation de sources du ZIP Wan2GP fourni :

La 0.2.1 intègre aussi les changements du pipeline de `Wan2GP-main(2).zip` (13.0), en conservant les adaptations de capture et de continuation. Les modules natifs VAE/transformer sont importés depuis l'installation utilisateur.

- `pipeline.py` : `models/minimax_h3/pipeline.py`, adapté aux points de capture/reprise ; imports internes redirigés vers Wan2GP.
- `packing.py` : `models/minimax_h3/components/packing.py`, avec ajout du placement audio par décalage numérique. Le marquage SPDX Apache-2.0 du fichier est conservé.
- `layout.py` : méthode `_layout` de `models/minimax_h3/transformer.py`, utilisée uniquement sur les instances prototype.
- Les quatre fichiers `defaults/` sont dérivés des définitions correspondantes du ZIP (noms et identifiants adaptés).

Le texte de licence fourni avec Wan2GP est conservé dans `LICENSE-Wan2GP.txt`. Les droits/licences des composants tiers restent applicables ; ce plugin ne relicencie pas leurs sources ou les poids. Aucun poids n'est distribué ici.

Les fichiers ajoutés pour l'intégration de ce prototype sont fournis sous les mêmes conditions que l'implémentation Wan2GP jointe, sans garantie supplémentaire de fonctionnement ou de qualité générative.

Aucun code des projets ComfyUI cités dans le README n'est inclus. Leurs mécanismes ont servi de références conceptuelles ; leurs licences ne sont pas étendues à ce plugin par ces citations.
