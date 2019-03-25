# Abstrakt
Jedním z problémů, s kterými se setkává autonomní řízení vozidel, je schopnost rozpoznat a klasifikovat dopravní značky. Pro účely počítačového vidění se v posledních letech významně rozšířili konvoluční neuronové sítě. Použití neuronových sítí ke komplexní klasifikaci dopravních značek částečně limituje nedostatek dat. Předmětem této práce tedy je sloučení existujících veřejně dostupných dat a vytvoření rozsáhlejší datové sady dopravních značek vyskytujících se v České republice. Tu následně využijeme k natrénování modelu schopného třídit české dopravní značky.

- rozšířit

# Úvod
Autonomní řízení vozidel vyžaduje pokročilou schopnost orientace v prostoru a adaptace na změny prostředí. Kromě rozpoznávání překážek, sledování jízdních pruhů a dalších úkolů, musí být samořídící vůz schopen také najít a rozpoznat dopravní značky v okolí. Zpravidla se k tomuto účelu používají přední kamery, které pravidelně snímají obraz před vozidlem a v reálném čase fotografie analyzují. Ukazuje se, že je vhodné rozdělit tento problém na dvě části. Nejdříve je třeba zjistit, zda se v získaném obrázku vůbec nějaká značka vyskytuje a pokud ano, pak dokázat určit kde a ideálně označit celý výřez, v kterém se nachází. Vyříznutá část obrázku se v následujícím kroku vyhodnotí a určí se, o kterou dopravní značku se jedná. V této práci se pokusíme přiblížit řešení druhé části.
Ke klasifikaci obrazu se v poslední době využívají velmi úspešně konvoluční neuronové sítě. Tato varianta neuronových sítí je schopna dávat do souvislosti sousední části obrazu, a tak v nižších vrstvách rozpoznávat hrany, ty ve vyžších vrstvách skládat v jednoduché tvary a z těch pak konečně rozpoznávat složitější objekty. Výhoda konvolučních sítí je také malá míra předzpracování dat, nižší počet trénovacích parametrů, z toho plynoucí rychlejší trénování a již zmíněná schopnost zachycení prostorové struktury. Fenoménem posledních let jsou velmi hluboké neuronové sítě skládající se z několika desítek až stovek vrstev. Nevýhodou těchto sítí je pomalá doba trénování a velké rozměry, které limitují jejich nasazení v real-time aplikacích. Naší snahou by proto mělo být nalezení nejmenšího takového modelu, který bude schopen uspokojivě značky klasifikovat.
Stejně jako jiné metody strojového učení i neuronové sítě potřebují k trénování velké množství kvalitních dat. Datových sad dopravních značek je volně dostupných několik. Některé však obsahují jen malé množství značek, málo různých kategorií a často se stejné značky liší mezi státy, v kterých byla data získána.
**Cílem této práce bude vybrání a sloučení existujících datových sad, jejich doplnění a vytvoření rozsáhlejší datové sady značek vyskytujících se v České republice a její nasledné využití k natrénování modelu schopného třídit české dopravní značky.**

- co to znamená uspokojivě?
- citace
- příliš odborné?
- hypotéza

# Related works
## Jiné přístupy
- SVM
- color-based
## Práce v tomto oboru
- zmínit AlexNet?
- state of the art
- jiné publikace (počty citací)

# Dataset
## Obecně o značkách
- specifika značek (jasné tvary a barvy)
- [úmluva o značkách](https://en.wikipedia.org/wiki/Vienna_Convention_on_Road_Signs_and_Signals)
## Použité datasety - obecné vlastnosti + které značky jsem vybral
- german
- belgian
- italian
- chinese
- telenav?
## Můj dataset
- jak a kde jsem ho vytvářel
## Finální dataset
- které značky jsem vybral - kompletní seznam v přiloze
- preprocessing
- augmentation

# Model
## Metodika
- jazyk & framework
- hyperparametry - jaké jsem zkoušel, jaké zafixoval a proč
## Výsledky
- grafy průběhu trénování (TensorBoard)
- který byl nejlepší

# Diskuze a závěr
- s čím měl model problémy
- vizualizace
- co by se dalo zlepšit, dodělat
