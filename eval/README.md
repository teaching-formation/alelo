# Banc de vérité d'alélo (« golden eval »)

But : **attraper une régression avant l'utilisateur**. Une erreur déjà connue ne doit jamais repasser.
On rejoue une batterie de questions-vérité contre l'API réelle (`/api/chat`) et on vérifie chaque
réponse par assertions simples (présence/absence de faits, langue). Vert = feu vert pour une démo.

## Lancer

```bash
# toute la batterie (Auto + Expert)
python3 eval/run_eval.py

# seulement le mode Auto (rapide) — ignore les cas Expert (14B, lents)
python3 eval/run_eval.py --mode auto

# une catégorie
python3 eval/run_eval.py --category ministres

# filtre par sous-chaîne (id ou catégorie)
python3 eval/run_eval.py -k president

# voir la réponse même quand ça passe
python3 eval/run_eval.py -v

# API sur une autre URL
python3 eval/run_eval.py --url http://localhost:8088
```

Code de sortie **0** si tout passe, **1** s'il y a au moins un échec (utilisable en garde-fou avant une démo/déploiement).
Rapports écrits : `eval/report.md` (lisible) et `eval/last_run.json` (machine).

## Auto-génération depuis les sources (exhaustivité)

Les faits **énumérables** ne s'écrivent pas à la main : ils se **dérivent** de la donnée qui fait autorité.

```bash
python3 eval/gen_golden.py            # (ré)écrit golden.json = golden_core.json + cas générés
python3 eval/gen_golden.py --dry-run  # affiche ce qui serait généré, sans écrire
python3 eval/gen_golden.py --services # ajoute aussi les cas services (graphe — à vérifier avant)
```

- `golden_core.json` — le **curé non-énumérable** : comptes/rangs, culture générale, honnêteté,
  langue, anti-esquive, cas experts. C'est là qu'on écrit à la main.
- `gen_golden.py` — génère les **35 membres du gouvernement** depuis
  `data/raw_gouvernement/composition.json` (un cas « qui est le <poste> ? » par ligne, nom complet
  extrait). Résultat fusionné avec le core → `golden.json`.
- **Conséquence clé** : après un remaniement, tu mets à jour `composition.json`, tu relances
  `gen_golden.py`, et la batterie est de nouveau exhaustive — zéro maintenance manuelle.
- Les cas **services** (institution par service) sont **opt-in** : le graphe est extrait par LLM,
  pas une vérité sûre — on ne les active qu'après avoir vérifié ses institutions.

`golden.json` est donc un **artefact généré** : ne l'édite pas à la main, édite `golden_core.json`
(non-énumérable) ou la source (`composition.json`) puis régénère.

## Ajouter une question-vérité (curé)

Une ligne dans `eval/golden_core.json` :

```json
{ "id": "min-sante", "category": "ministres", "mode": "auto",
  "q": "qui est le ministre de la Santé ?",
  "expect_any": ["dimba"], "forbid": ["aka aouele"] }
```

Champs :
- `mode` : `"auto"` (7B) ou `"expert"` (14B agentique).
- `expect_all` : liste d'exigences — chaîne (doit apparaître) **ou** liste (au moins une de la liste).
- `expect_any` : au moins une des chaînes doit apparaître.
- `forbid` : aucune de ces chaînes (en plus de l'anti-esquive global du lanceur).
- `expect_lang` : `"fr"` ou `"en"` — vérifie la langue de la réponse (stickiness).

Comparaison **insensible à la casse et aux accents**, apostrophes/traits d'union → espaces.
Écrire les valeurs attendues déjà « normalisées » (ex. `"n guessan"` et non `"N'Guessan"`).

## Règle d'or

Chaque fois qu'une **mauvaise surprise** est corrigée (👎, bug en démo), on **ajoute une question-vérité**
qui l'aurait attrapée. La batterie ne fait que grandir : la couverture devient la mémoire anti-régression.

## Les 5 familles couvertes

1. **composition** — comptes et rangs du gouvernement (34 membres, 1 vice-PM…).
2. **ministres** — les 29 titulaires (fraîcheur : numérique, salubrité, santé, agriculture…).
3. **routage** — le bon service → la bonne institution (passeport→SNEDAI, CNI→ONECI…).
4. **culture_generale / honnetete** — répond au GK ; avoue honnêtement quand il ne sait pas.
5. **langue / anti_esquive / expert** — stickiness FR/EN, jamais « consultez le site », agentique multi-volet.
