# %%
'''
This script creates augmented Uspanteko data by combining morphemes according to one of two morpheme templates: transitive and intransitive verbs. 

The transitive morpheme template requires one of each of the following markers in addition to the verb itself: TAM, absolutive, ergative, and suffix. 
The intransitive morpheme template requires one of each of the following markers in addition to the verb itself: TAM, absolutive, and suffix. 

The script outputs IGT glosses formatted like the example below to two .txt files:
\t xatq'alajínik
\p COM-A2S-E1P-VT-SC
\g COM-A2S-E1P-aparecer-SC
\l Volviste a aparecer.

'''

# %%
import re
import mlconjug3 # Install using requirements.txt file
from mlconjug3 import Conjugator

# %%
conjugator = Conjugator(language='es')  # Instantiate Spanish conjugator 

# %%
# The following section contains the dictionaries for each morpheme "building block". 
# Each dictionary consists of a list of three elements used to construct the IGT gloss. 
# The elements for each list will be listed in a comment above each building block. 

# [Uspanteko morpheme, Glossing abbrev., Name of Spanish verb tense]
tam = {'x': ['x', 'COM', 'Indicativo pretérito perfecto simple'], 
       't': ['t', 'INC', 'Indicativo presente']}

# [Uspanteko morpheme, Glossing abbrev., Pronoun in Spanish]
absolutives = {'A1S': ['in', 'A1S', 'yo'], 
               'A2S': ['at', 'A2S', 'tú'], 
               'A3S': ['', 'A3S', 'él'], 
               'A1P': ['oj', 'A1P', 'nosotros'],
               'A2P': ['at', 'A2S', 'ellos'], 
               'A3P': ['', 'A3S', 'ellos']}

# [Uspanteko morpheme, Glossing abbrev., Pronoun in Spanish]
ergatives = {'E1S': ['in', 'E1S', 'yo'], 
             'E2S': ['a', 'E2S', 'tú'], 
             'E3S': ['j', 'E3S', 'él'],
             'E1P': ['qa', 'E1P', 'nosotros'],
             'E2P': ['a',  'E2S', 'ellos'],
             'E3P': ['j',  'E3S', 'ellos']}

# [Verb root, Glossing abbrev., Verb in Spanish]
transitive_verbs = {'k\'iliik': ['k\'ili', 'VT', 'tostar'], 
                    'k\'isiik': ['k\'isi', 'VT', 'terminar'],
                    'xutiik': ['xuti', 'VT', 'abandonar'],
                    'tzaqónik': ['tzaqón', 'VT', 'abortar'],
                    'job\'jób\'ik': ['job\'jób\'', 'VT', 'aboyar'],
                    'kojónik': ['kojón', 'VT', 'aceptar'],
                    'pixib\'anik': ['pixib\'an', 'VT', 'aconsejar'],
                    'q\'alúnik': ['q\'alún', 'VT', 'abrazar'],
                    'pechínik': ['pechín', 'VT', 'acompañar'],
                    'jaqiik': ['jaqi', 'VT', 'abrir'],
                    'wersánik': ['wersán', 'VT', 'adormecer'],
                    'tyoxínik': ['tyoxín', 'VT', 'agradecer'],
                    'chob\'iik': ['chob\'i', 'VT', 'agujerear'],
                    'jiq\'sánik': ['jiq\'sán', 'VT', 'ahogar'],
                    'nq\'asájik': ['nq\'asáj', 'VT', 'alcanzar'],
                    'jib\'ib\'íjik': ['jib\'ib\'íj', 'VT', 'chicotear'],
                    'pich\'újik': ['pich\'új', 'VT', 'analizar'],
                    'tosiik': ['tosi', 'VT', 'apartar'],
                    't\'okiik': ['t\'oki', 'VT', 'apretar'],
                    'awsíjik': ['awsíj', 'VT', 'bendecir'],
                    'tz\'ub\'ánik': ['tz\'ub\'án', 'VT', 'besar'],
                    'q\'ab\'iik': ['q\'ab\'i', 'VT', 'calumniar'],
                    'ch\'elénik': ['ch\'elén', 'VT', 'cargar'],
                    'b\'ukiik': ['b\'uki', 'VT', 'cobijar'],
                    'koliik': ['koli', 'VT', 'defender'],
                    'mayiik': ['mayi', 'VT', 'detener'],
                    'chiqánik': ['chiqá', 'VT', 'empujar'],
                    'ta\'iik': ['ta\'i', 'VT', 'encontrar'],
                    'k\'utiik': ['k\'uti', 'VT', 'enseñar'],
                    'tzab\'ánik': ['tzab\'án', 'VT', 'estorbar'],
                    'b\'isiik': ['b\'isi', 'VT', 'extrañar'],
                    'b\'itiik': ['b\'iti', 'VT', 'guardar'],
                    'k\'amiik': ['k\'ami',  'VT', 'guiar'],
                    'itz\'b\'énik': ['itz\'b\'én', 'VT', 'jugar'],
                    'raq\'iik': ['raq\'i', 'VT', 'lamer'],
                    'b\'aqiik': ['b\'aqi', 'VT', 'lavar'],
                    'sik\'ijiik': ['sik\'iji', 'VT', 'llamar'],
                    'k\'achínik': ['k\'achín', 'VT', 'molestar'],
                    'kach\'iik': ['kach\'i', 'VT', 'morder'],
                    'k\'eriik': ['k\'eri', 'VT', 'partir'],
                    'ch\'o\'jánik': ['ch\'o\'ján', 'VT', 'pelear'],
                    'qejiik': ['qeji', 'VT', 'prestar'],
                    'nuqiik': ['nuqi', 'VT', 'proteger'],
                    'josiik': ['josi', 'VT', 'raspar'],
                    'ta\'jájík': ['ta\'jáj', 'VT', 'reclamar'],
                    'sipánik': ['sipán', 'VT', 'regalar'],
                    'kolónik': ['kolón', 'VT', 'rescatar'],
                    'walqatinik': ['walqatin', 'VT', 'revolcar'],
                    'alq\'ánik': ['alq\'án', 'VT', 'robar'],
                    'b\'ujiik': ['b\'uji', 'VT', 'somatar'],
                    'pok\'xínik': ['pok\'xín', 'VT', 'taconear'],
                    'ch\'uqiik': ['ch\'uqi', 'VT', 'tapar'],
                    'solínik': ['solín', 'VT', 'visitar'],
                    'tzalqomíjik': ['tzalqomíj', 'VT', 'voltear'],
                    }

# [Verb root, Glossing abbrev., Verb in Spanish]
intransitive_verbs = {'k\'iyiik': ['k\'iyi', 'VI', 'crecer'], 
                    'paxínik': ['paxín', 'VI', 'abundar'],
                    'nimajiik': ['nimaji', 'VI', 'aceptar'],
                    'koch\'iik': ['koch\'i', 'VI', 'aguantar'],
                    'chupxínik': ['chupxín', 'VI', 'alumbrar'],
                    'qejiik': ['qeji', 'VI', 'bajar'],
                    'atínik': ['atín', 'VI', 'bañar'],
                    'saysapúnik': ['saysapún', 'VI', 'bracear'],
                    'xik\'ánik': ['xik\'án', 'VI', 'brincar'],
                    'ch\'erxínik': ['ch\'erxín', 'VI', 'cacarear'],
                    'kosiik': ['kosi', 'VI', 'cansar'],
                    'nach\'xínik': ['nach\'xín', 'VI', 'comer'],
                    'rayínik': ['rayín', 'VI', 'desear'],
                    'okiik': ['oki', 'VI', 'entrar'],
                    'lakánik': ['lakán', 'VI', 'gatear'],
                    'tzawxínik': ['tzawxín', 'VI', 'gritar'],
                    'wet\'et\'ik':['wet\'et\'', 'VI', 'hablar'],
                    'b\'anawik':['b\'anaw', 'VI', 'hacer'],
                    'k\'isi\'ik': ['k\'isi\'', 'VI', 'nacer'],
                    'wo\'kotiik': ['wo\'koti', 'VI', 'pasear'],
                    'ch\'ob\'ólik': ['ch\'ob\'ól', 'VI', 'pensar'],
                    'menepúnik': ['menepún', 'VI', 'tartamudear'],
                    'b\'iríwik': ['b\'iríw', 'VI', 'temblar'],
                    'k\'isi\'ik': ['k\'isi\'', 'VI', 'vivir'],
                    }

# [Uspanteko morpheme, Glossing abbrev., Verb type]
suffixes = {'j': ['j', 'SC', 'transitive'], 
            'ik': ['ik', 'SC', 'intransitive']} 

# %%
# List of direct object pronouns for use in transitive Spanish translations
spanish_do = {'yo':'me', 
              'tú':'te', 
              'él':'lo', 
              'nosotros': 'nos', 
              'ellos': 'los',
              'ellos':'los'}

# %%
# Intransitive Verbs
# Morpheme Template: TAM + Absolutive + Verb + Suffix

with open('./Method2/Generated Data/intransitive_examples_unseg.txt', 'w') as f:
    for iverb in intransitive_verbs:
        conjugations = conjugator.conjugate(intransitive_verbs[iverb][2])
        for s in suffixes:
            if suffixes[s][2] == 'intransitive':
                for t in tam:
                    for abs in absolutives:
                        if absolutives[abs][1] == 'A2S' and absolutives[abs][2] == "ellos":
                            f.write(f"""
                                    \\t  {tam[t][0]}{absolutives[abs][0]}{intransitive_verbs[iverb][0]}{suffixes[s][0]} taq
                                    \\p  {tam[t][1]}-{absolutives[abs][1]}-{intransitive_verbs[iverb][1]}-{suffixes[s][1]} PL
                                    \\g  {tam[t][1]}-{absolutives[abs][1]}-{intransitive_verbs[iverb][2]}-{suffixes[s][1]} PL
                                    \\l  {conjugations['Indicativo', tam[t][2], absolutives[abs][2]]}
                                """)
                        elif absolutives[abs][1] == 'A2S' and absolutives[abs][2] != "ellos":
                            f.write(f"""
                                    \\t  {tam[t][0]}{absolutives[abs][0]}{intransitive_verbs[iverb][0]}{suffixes[s][0]}
                                    \\p  {tam[t][1]}-{absolutives[abs][1]}-{intransitive_verbs[iverb][1]}-{suffixes[s][1]}
                                    \\g  {tam[t][1]}-{absolutives[abs][1]}-{intransitive_verbs[iverb][2]}-{suffixes[s][1]}
                                    \\l  {conjugations['Indicativo', tam[t][2], absolutives[abs][2]]}
                                """)
                        elif  absolutives[abs][1] == 'A3S' and absolutives[abs][2] == "ellos":
                            f.write(f"""
                                    \\t  {tam[t][0]}{intransitive_verbs[iverb][0]}{suffixes[s][0]} taq
                                    \\p  {tam[t][1]}-{intransitive_verbs[iverb][1]}-{suffixes[s][1]} PL
                                    \\g  {tam[t][1]}-{intransitive_verbs[iverb][2]}-{suffixes[s][1]} PL
                                    \\l  {conjugations['Indicativo', tam[t][2], absolutives[abs][2]]}
                                """)
                        elif  absolutives[abs][1] == 'A3S' and absolutives[abs][2] != "ellos":
                            f.write(f"""
                                    \\t  {tam[t][0]}{intransitive_verbs[iverb][0]}{suffixes[s][0]} 
                                    \\p  {tam[t][1]}-{intransitive_verbs[iverb][1]}-{suffixes[s][1]} 
                                    \\g  {tam[t][1]}-{intransitive_verbs[iverb][2]}-{suffixes[s][1]}
                                    \\l  {conjugations['Indicativo', tam[t][2], absolutives[abs][2]]}
                                """)
                        else:
                            f.write(f"""
                                    \\t  {tam[t][0]}{absolutives[abs][0]}{intransitive_verbs[iverb][0]}{suffixes[s][0]}
                                    \\p  {tam[t][1]}-{absolutives[abs][1]}-{intransitive_verbs[iverb][1]}-{suffixes[s][1]}
                                    \\g  {tam[t][1]}-{absolutives[abs][1]}-{intransitive_verbs[iverb][2]}-{suffixes[s][1]}
                                    \\l  {conjugations['Indicativo', tam[t][2], absolutives[abs][2]]}
                                """)
            


# %%
# Transitive Verbs
# Morpheme Template: TAM + Absolutive + Ergative + Verb + Suffix

with open('./Method2/Generated Data/transitive_examples_unseg.txt', 'w') as file:
    for tverb in transitive_verbs:
        conjugations = conjugator.conjugate(transitive_verbs[tverb][2])
        for s in suffixes:
            if suffixes[s][2] == 'transitive':
                for t in tam:
                    for abs in absolutives:
                        for erg in ergatives:
                            if absolutives[abs][1] == 'A2S' and absolutives[abs][2] == "ellos":
                                if ergatives[erg][2] == "ellos":
                                    file.write(f"""
                                            \\t  {tam[t][0]}{absolutives[abs][0]}{ergatives[erg][0]}{transitive_verbs[tverb][0]}{suffixes[s][0]} taq taq
                                            \\p  {tam[t][1]}-{absolutives[abs][1]}-{ergatives[erg][1]}-{transitive_verbs[tverb][1]}-{suffixes[s][1]} PL PL
                                            \\g  {tam[t][1]}-{absolutives[abs][1]}-{ergatives[erg][1]}-{transitive_verbs[tverb][2]}-{suffixes[s][1]} PL PL
                                            \\l  {spanish_do[ergatives[erg][2]]} {conjugations['Indicativo', tam[t][2], absolutives[abs][2]]}
                                        """)
                                else:
                                    file.write(f"""
                                            \\t  {tam[t][0]}{absolutives[abs][0]}{ergatives[erg][0]}{transitive_verbs[tverb][0]}{suffixes[s][0]} taq
                                            \\p  {tam[t][1]}-{absolutives[abs][1]}-{ergatives[erg][1]}-{transitive_verbs[tverb][1]}-{suffixes[s][1]} PL
                                            \\g  {tam[t][1]}-{absolutives[abs][1]}-{ergatives[erg][1]}-{transitive_verbs[tverb][2]}-{suffixes[s][1]} PL
                                            \\l  {spanish_do[ergatives[erg][2]]} {conjugations['Indicativo', tam[t][2], absolutives[abs][2]]}
                                        """)
                            elif absolutives[abs][1] == 'A3S' and absolutives[abs][2] == "ellos":
                                if ergatives[erg][2] == "ellos":
                                    file.write(f"""
                                            \\t  {tam[t][0]}{ergatives[erg][0]}{transitive_verbs[tverb][0]}{suffixes[s][0]} taq taq
                                            \\p  {tam[t][1]}-{ergatives[erg][1]}-{transitive_verbs[tverb][1]}-{suffixes[s][1]} PL PL
                                            \\g  {tam[t][1]}-{ergatives[erg][1]}-{transitive_verbs[tverb][2]}-{suffixes[s][1]} PL PL
                                            \\l  {spanish_do[ergatives[erg][2]]} {conjugations['Indicativo', tam[t][2], absolutives[abs][2]]}
                                        """)
                                else:
                                    file.write(f"""
                                            \\t  {tam[t][0]}{ergatives[erg][0]}{transitive_verbs[tverb][0]}{suffixes[s][0]} taq
                                            \\p  {tam[t][1]}-{ergatives[erg][1]}-{transitive_verbs[tverb][1]}-{suffixes[s][1]} PL
                                            \\g  {tam[t][1]}-{ergatives[erg][1]}-{transitive_verbs[tverb][2]}-{suffixes[s][1]} PL
                                            \\l  {spanish_do[ergatives[erg][2]]} {conjugations['Indicativo', tam[t][2], absolutives[abs][2]]}
                                        """)
                            elif absolutives[abs][1] == 'A3S' and absolutives[abs][2] != "ellos":
                                    file.write(f"""
                                            \\t  {tam[t][0]}{ergatives[erg][0]}{transitive_verbs[tverb][0]}{suffixes[s][0]}
                                            \\p  {tam[t][1]}-{ergatives[erg][1]}-{transitive_verbs[tverb][1]}-{suffixes[s][1]}
                                            \\g  {tam[t][1]}{ergatives[erg][1]}-{transitive_verbs[tverb][2]}-{suffixes[s][1]}
                                            \\l  {spanish_do[absolutives[abs][2]]} {conjugations['Indicativo', tam[t][2], ergatives[erg][2]]}
                                        """)
                            else:
                                    file.write(f"""
                                            \\t  {tam[t][0]}{absolutives[abs][0]}{ergatives[erg][0]}{transitive_verbs[tverb][0]}{suffixes[s][0]}
                                            \\p  {tam[t][1]}-{absolutives[abs][1]}-{ergatives[erg][1]}-{transitive_verbs[tverb][1]}-{suffixes[s][1]}
                                            \\g  {tam[t][1]}-{absolutives[abs][1]}-{ergatives[erg][1]}-{transitive_verbs[tverb][2]}-{suffixes[s][1]}
                                            \\l  {spanish_do[absolutives[abs][2]]} {conjugations['Indicativo', tam[t][2], ergatives[erg][2]]}
                                        """)
                                
                


