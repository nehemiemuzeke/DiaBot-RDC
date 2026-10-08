"""
================================================================================
FICHIER : multilingual.py
RESPONSABILITÉ : Système de traduction, localisation et détection de langue
================================================================================
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config import (
    LangueSupportee, LANGUE_PAR_DEFAUT, LANGUES_INFO,
    TRANSLATIONS_DIR,
)

logger = logging.getLogger(__name__)


# ============================================================================
# STRUCTURES DE DONNÉES
# ============================================================================

@dataclass
class TraductionResultat:
    """Résultat d'une traduction ou détection de langue."""
    texte: str
    langue_detectee: str
    confiance: float
    code_switching: bool = False
    langues_mixees: list[str] = field(default_factory=list)
    fallback_utilise: bool = False


@dataclass
class MessageLocalise:
    """Message localisé avec métadonnées culturelles."""
    texte: str
    langue: str
    ton: str = "neutre"
    contexte_culturel: Optional[str] = None


# ============================================================================
# DICTIONNAIRES DE TRADUCTION COMPLETS
# ============================================================================

UI_TRADUCTIONS: dict[str, dict[str, str]] = {
    "fr": {
        "app_title": "DiaBot-RDC",
        "welcome": "Bienvenue sur DiaBot-RDC",
        "login": "Se connecter",
        "register": "S'inscrire",
        "logout": "Se déconnecter",
        "profile": "Mon profil",
        "settings": "Paramètres",
        "dashboard": "Tableau de bord",
        "add_measurement": "Ajouter une mesure",
        "history": "Historique",
        "statistics": "Statistiques",
        "reports": "Rapports",
        "chat": "Discuter avec DiaBot",
        "qr_code": "Mon QR Code",
        "scan_qr": "Scanner un QR",
        "import_data": "Importer des données",
        "export_data": "Exporter mes données",
        "alerts": "Alertes",
        "emergency": "Urgence",
        "save": "Enregistrer",
        "cancel": "Annuler",
        "confirm": "Confirmer",
        "delete": "Supprimer",
        "edit": "Modifier",
        "close": "Fermer",
        "back": "Retour",
        "next": "Suivant",
        "loading": "Chargement...",
        "error": "Erreur",
        "success": "Succès",
        "blood_sugar": "Glycémie",
        "fasting": "À jeun",
        "after_meal": "Après repas",
        "before_meal": "Avant repas",
        "bedtime": "Au coucher",
        "night": "Nuit",
        "morning": "Matin",
        "afternoon": "Après-midi",
        "evening": "Soir",
        "today": "Aujourd'hui",
        "yesterday": "Hier",
        "this_week": "Cette semaine",
        "this_month": "Ce mois",
        "last_3_months": "3 derniers mois",
        "doctor_mode": "Mode médecin",
        "patient_mode": "Mode patient",
        "language": "Langue",
        "notifications": "Notifications",
        "privacy": "Confidentialité",
        "help": "Aide",
        "about": "À propos",
        "consent": "Consentement",
        "i_agree": "J'accepte",
        "i_decline": "Je refuse",
        "measurement_time": "Heure de la mesure",
        "measurement_value": "Valeur de la mesure",
        "unit": "Unité",
        "notes": "Notes",
        "meal": "Repas",
        "physical_activity": "Activité physique",
        "stress_level": "Niveau de stress",
        "medication": "Médicament",
        "permission_level": "Niveau de permission",
        "qr_expires_in": "Le QR expire dans",
        "minutes": "minutes",
        "generate_new_qr": "Générer un nouveau QR",
        "revoke_qr": "Révoquer le QR",
        "share_with_doctor": "Partager avec mon médecin",
    },
    "ln": {
        "app_title": "DiaBot-RDC",
        "welcome": "Boyei malamu na DiaBot-RDC",
        "login": "Kota",
        "register": "Komisa nkombo",
        "logout": "Bima",
        "profile": "Profil na ngai",
        "settings": "Mibeko",
        "dashboard": "Etando ya mosala",
        "add_measurement": "Bakisa bomeki",
        "history": "Makambo ya kala",
        "statistics": "Bilanga",
        "reports": "Mikanda",
        "chat": "Solola na DiaBot",
        "qr_code": "QR Code na ngai",
        "scan_qr": "Tala QR",
        "import_data": "Kotisa ba données",
        "export_data": "Bimisa ba données na ngai",
        "alerts": "Makebisi",
        "emergency": "Likama",
        "save": "Bomba",
        "cancel": "Tika",
        "confirm": "Ndima",
        "delete": "Longola",
        "edit": "Bongola",
        "close": "Kanga",
        "back": "Zonga",
        "next": "Oyo elandi",
        "loading": "Ezali kokende...",
        "error": "Libunga",
        "success": "Elamu",
        "blood_sugar": "Sukali ya makila",
        "fasting": "Nzala",
        "after_meal": "Nsima ya kolia",
        "before_meal": "Liboso ya kolia",
        "bedtime": "Ntango ya kolala",
        "night": "Butu",
        "morning": "Ntongo",
        "afternoon": "Nzanga",
        "evening": "Mpokwa",
        "today": "Lelo",
        "yesterday": "Lobi eleki",
        "this_week": "Poso oyo",
        "this_month": "Sanza oyo",
        "last_3_months": "Basanza 3 ya nsuka",
        "doctor_mode": "Mode ya monganga",
        "patient_mode": "Mode ya mokonzi",
        "language": "Monoko",
        "notifications": "Biyebisi",
        "privacy": "Bomba makambo",
        "help": "Lisalisi",
        "about": "Mpo na biso",
        "consent": "Boyokani",
        "i_agree": "Nandimi",
        "i_decline": "Naboyi",
        "measurement_time": "Ngonga ya bomeki",
        "measurement_value": "Motuya ya bomeki",
        "unit": "Eteni",
        "notes": "Makomi",
        "meal": "Bilei",
        "physical_activity": "Mosala ya nzoto",
        "stress_level": "Motungisi",
        "medication": "Nkisi",
        "permission_level": "Ndingisa",
        "qr_expires_in": "QR ekosila na",
        "minutes": "miniti",
        "generate_new_qr": "Sala QR ya sika",
        "revoke_qr": "Longola QR",
        "share_with_doctor": "Kabela monganga na ngai",
    },
    "sw": {
        "app_title": "DiaBot-RDC",
        "welcome": "Karibu kwenye DiaBot-RDC",
        "login": "Ingia",
        "register": "Jisajili",
        "logout": "Toka",
        "profile": "Wasifu wangu",
        "settings": "Mipangilio",
        "dashboard": "Dashibodi",
        "add_measurement": "Ongeza kipimo",
        "history": "Historia",
        "statistics": "Takwimu",
        "reports": "Ripoti",
        "chat": "Ongea na DiaBot",
        "qr_code": "QR Code yangu",
        "scan_qr": "Skeni QR",
        "import_data": "Ingiza data",
        "export_data": "Toa data yangu",
        "alerts": "Tahadhari",
        "emergency": "Dharura",
        "save": "Hifadhi",
        "cancel": "Ghairi",
        "confirm": "Thibitisha",
        "delete": "Futa",
        "edit": "Hariri",
        "close": "Funga",
        "back": "Rudi",
        "next": "Inayofuata",
        "loading": "Inapakia...",
        "error": "Kosa",
        "success": "Mafanikio",
        "blood_sugar": "Sukari ya damu",
        "fasting": "Ufungo",
        "after_meal": "Baada ya chakula",
        "before_meal": "Kabla ya chakula",
        "bedtime": "Wakati wa kulala",
        "night": "Usiku",
        "morning": "Asubuhi",
        "afternoon": "Mchana",
        "evening": "Jioni",
        "today": "Leo",
        "yesterday": "Jana",
        "this_week": "Wiki hii",
        "this_month": "Mwezi huu",
        "last_3_months": "Miezi 3 iliyopita",
        "doctor_mode": "Hali ya daktari",
        "patient_mode": "Hali ya mgonjwa",
        "language": "Lugha",
        "notifications": "Arifa",
        "privacy": "Faragha",
        "help": "Msaada",
        "about": "Kuhusu",
        "consent": "Idhini",
        "i_agree": "Nakubali",
        "i_decline": "Nakataa",
        "measurement_time": "Wakati wa kipimo",
        "measurement_value": "Thamani ya kipimo",
        "unit": "Kipimo",
        "notes": "Maelezo",
        "meal": "Chakula",
        "physical_activity": "Shughuli za mwili",
        "stress_level": "Kiwango cha msongo",
        "medication": "Dawa",
        "permission_level": "Kiwango cha ruhusa",
        "qr_expires_in": "QR itaisha ndani ya",
        "minutes": "dakika",
        "generate_new_qr": "Tengeneza QR mpya",
        "revoke_qr": "Batilisha QR",
        "share_with_doctor": "Shiriki na daktari wangu",
    },
    "lu": {
        "app_title": "DiaBot-RDC",
        "welcome": "Tukutambula ne DiaBot-RDC",
        "login": "Ingila",
        "register": "Dileja dijina",
        "logout": "Fuma",
        "profile": "Dipatshilu dia meme",
        "settings": "Mikenji",
        "dashboard": "Tshisalu",
        "add_measurement": "Bika tshipimilu",
        "history": "Bintu bya kale",
        "statistics": "Bilanga",
        "reports": "Mikanda",
        "chat": "Amba ne DiaBot",
        "qr_code": "QR Code wa meme",
        "scan_qr": "Tala QR",
        "import_data": "Ingisha bintu",
        "export_data": "Bimisha bintu bia meme",
        "alerts": "Makenji",
        "emergency": "Bweshi",
        "save": "Bika",
        "cancel": "Leka",
        "confirm": "Tangila",
        "delete": "Futa",
        "edit": "Sokolola",
        "close": "Kanga",
        "back": "Buya",
        "next": "Tshikwawu",
        "loading": "Kudi kwenda...",
        "error": "Difua",
        "success": "Bwino",
        "blood_sugar": "Sukadi ya mashi",
        "fasting": "Nzala",
        "after_meal": "Pantu wa kudia",
        "before_meal": "Kudi kudia",
        "bedtime": "Tshikulu tsha kulala",
        "night": "Bufuku",
        "morning": "Luseke",
        "afternoon": "Mutshikulu",
        "evening": "Dilolo",
        "today": "Lelu",
        "yesterday": "Maloba",
        "this_week": "Lusuku luno",
        "this_month": "Mweshi uno",
        "last_3_months": "Myeshi 3 ya ku mpela",
        "doctor_mode": "Mukenji wa muganga",
        "patient_mode": "Mukenji wa mulwadi",
        "language": "Ludimi",
        "notifications": "Bimanyishi",
        "privacy": "Bisungu",
        "help": "Lusadilu",
        "about": "Pa tuetu",
        "consent": "Tshitangilu",
        "i_agree": "Natangile",
        "i_decline": "Nakane",
        "measurement_time": "Tshikulu tsha tshipimilu",
        "measurement_value": "Bunene bwa tshipimilu",
        "unit": "Tshilema",
        "notes": "Makomi",
        "meal": "Bidia",
        "physical_activity": "Misalu ya mubidi",
        "stress_level": "Bunene bwa mpasi",
        "medication": "Mushipu",
        "permission_level": "Ndingisha",
        "qr_expires_in": "QR ukufua mu",
        "minutes": "miniti",
        "generate_new_qr": "Sala QR wa mushimbi",
        "revoke_qr": "Futa QR",
        "share_with_doctor": "Kabela muganga wa meme",
    },
    "kg": {
        "app_title": "DiaBot-RDC",
        "welcome": "Tombi na DiaBot-RDC",
        "login": "Kota",
        "register": "Soneka zina",
        "logout": "Katuka",
        "profile": "Profile na mono",
        "settings": "Mansangu",
        "dashboard": "Kisalu",
        "add_measurement": "Yika ntalu",
        "history": "Mambu ma ntama",
        "statistics": "Bilanga",
        "reports": "Mikanda",
        "chat": "Solola na DiaBot",
        "qr_code": "QR Code na mono",
        "scan_qr": "Tala QR",
        "import_data": "Kota ba données",
        "export_data": "Bimisa ba données na mono",
        "alerts": "Makabu",
        "emergency": "Mpasi",
        "save": "Bumba",
        "cancel": "Yambula",
        "confirm": "Ndima",
        "delete": "Katula",
        "edit": "Soba",
        "close": "Kanga",
        "back": "Vutuka",
        "next": "Yina ke landa",
        "loading": "Ke kwenda...",
        "error": "Foti",
        "success": "Mbote",
        "blood_sugar": "Sukadi ya menga",
        "fasting": "Nzala",
        "after_meal": "Na nima ya dia",
        "before_meal": "Na ntwala ya dia",
        "bedtime": "Ntangu ya lala",
        "night": "Mpimpa",
        "morning": "Ntangu",
        "afternoon": "Nkutu",
        "evening": "Nkokila",
        "today": "Bubu",
        "yesterday": "Zuzi",
        "this_week": "Mposo yayi",
        "this_month": "Ngonda yayi",
        "last_3_months": "Bangonda 3 ya nsuka",
        "doctor_mode": "Mode ya nganga",
        "patient_mode": "Mode ya mbefo",
        "language": "Ndinga",
        "notifications": "Bisangu",
        "privacy": "Bumbi",
        "help": "Lusadisu",
        "about": "Mambu ma beto",
        "consent": "Ndima",
        "i_agree": "Mondimidi",
        "i_decline": "Mondimidi ko",
        "measurement_time": "Ntangu ya ntalu",
        "measurement_value": "Ntalu",
        "unit": "Kiteso",
        "notes": "Mikomi",
        "meal": "Madia",
        "physical_activity": "Misalu ya nitu",
        "stress_level": "Ntalu ya mpasi",
        "medication": "Nkisi",
        "permission_level": "Ndingisa",
        "qr_expires_in": "QR ke mana na",
        "minutes": "miniti",
        "generate_new_qr": "Sala QR ya mpa",
        "revoke_qr": "Katula QR",
        "share_with_doctor": "Kabela nganga na mono",
    },
}

IA_MESSAGES: dict[str, dict[str, str]] = {
    "fr": {
        "greeting": "Bonjour ! Je suis DiaBot-RDC, votre compagnon pour le suivi de votre diabète. Comment allez-vous aujourd'hui ?",
        "greeting_evening": "Bonsoir ! Comment s'est passée votre journée ? Avez-vous pris vos mesures ?",
        "ask_measurement": "Avez-vous mesuré votre glycémie aujourd'hui ? N'oubliez pas, c'est important pour votre santé !",
        "measurement_recorded": "✅ Votre glycémie de {valeur} g/L a bien été enregistrée. {commentaire}",
        "measurement_normal": "🟢 Votre glycémie est dans la zone normale. Continuez comme ça, vous faites du bon travail !",
        "measurement_hypo_mild": "🟡 Votre glycémie est un peu basse ({valeur} g/L). Mangez un fruit ou un peu de sucre et remesurez dans 15 minutes.",
        "measurement_hypo_severe": "🔴 Attention ! Votre glycémie est très basse ({valeur} g/L). Prenez immédiatement du sucre (jus, bonbon, miel) et contactez quelqu'un autour de vous. Si ça ne remonte pas, allez à l'hôpital.",
        "measurement_hyper_mild": "🟡 Votre glycémie est un peu élevée ({valeur} g/L). Buvez de l'eau, faites une petite marche si possible, et vérifiez dans 2 heures.",
        "measurement_hyper_severe": "🔴 Votre glycémie est très élevée ({valeur} g/L). Buvez beaucoup d'eau, évitez les aliments sucrés, et consultez votre médecin rapidement.",
        "measurement_urgence": "🚨 URGENCE ! Votre glycémie est à {valeur} g/L. C'est un niveau dangereux. Rendez-vous IMMÉDIATEMENT à l'hôpital le plus proche ou appelez le {numero_urgence}.",
        "encouragement": "Bravo pour votre régularité ! Le suivi quotidien est la clé d'une bonne santé. 💪",
        "tip_hydration": "💧 Conseil du jour : Buvez au moins 8 verres d'eau par jour. L'hydratation aide à réguler la glycémie.",
        "tip_activity": "🚶 Conseil du jour : Une marche de 30 minutes après le repas aide à faire baisser la glycémie naturellement.",
        "tip_nutrition": "🥗 Conseil du jour : Préférez le fufu de manioc au fufu de maïs, il a un index glycémique plus bas. Ajoutez du pondu (feuilles de manioc) riche en fibres !",
        "cannot_diagnose": "Je comprends votre inquiétude, mais je ne suis pas médecin et je ne peux pas poser de diagnostic. Je vous recommande vivement de consulter votre médecin ou d'aller au centre de santé le plus proche.",
        "cannot_prescribe": "Je ne peux pas vous prescrire de médicaments ni modifier votre traitement. Seul votre médecin peut le faire. Continuez à prendre vos médicaments comme prescrit.",
        "goodbye": "Au revoir et prenez bien soin de vous ! N'hésitez pas à revenir si vous avez des questions. 🌟",
    },
    "ln": {
        "greeting": "Mbote ! Ngai nazali DiaBot-RDC, moninga na yo mpo na kolanda sukadi na yo. Ozali malamu lelo ?",
        "greeting_evening": "Mpokwa malamu ! Mokolo na yo elekaki ndenge nini ? Omemaki bomeki na yo ?",
        "ask_measurement": "Omemaki sukadi ya makila na yo lelo ? Kobosana te, ezali na ntina mpo na kolongono na yo !",
        "measurement_recorded": "✅ Sukadi na yo ya {valeur} g/L ebombami malamu. {commentaire}",
        "measurement_normal": "🟢 Sukadi na yo ezali na esika ya malamu. Kobá bongo, ozali kosala malamu !",
        "measurement_hypo_mild": "🟡 Sukadi na yo ekiti moke ({valeur} g/L). Lia mbuma moko to sukadi moke mpe mema lisusu nsima ya miniti 15.",
        "measurement_hypo_severe": "🔴 Keba ! Sukadi na yo ekiti mingi ({valeur} g/L). Mela sukadi nokinoki (mai ya mbuma, bonbon, nzoi) mpe bengá moto moko pene na yo. Soki emati te, kende lopitalo.",
        "measurement_hyper_mild": "🟡 Sukadi na yo emati moke ({valeur} g/L). Mela mai mingi, tambola moke soki okoki, mpe tala lisusu nsima ya ngonga 2.",
        "measurement_hyper_severe": "🔴 Sukadi na yo emati mingi ({valeur} g/L). Mela mai mingi, boya bilei ya sukadi, mpe kende epai ya monganga nokinoki.",
        "measurement_urgence": "🚨 LIKAMA ! Sukadi na yo ezali na {valeur} g/L. Ezali likama. Kende LOPITALO MBALA MOKO to bengá {numero_urgence}.",
        "cannot_diagnose": "Nayoki mpasi na yo, kasi nazali monganga te mpe nakoki koyebisa yo maladi nini ozali na yango. Kende epai ya monganga.",
        "cannot_prescribe": "Nakoki kopesa yo nkisi te to kobongola traitement na yo. Monganga na yo nde akoki kosala yango.",
        "goodbye": "Tikala malamu mpe batela nzoto na yo ! 🌟",
    },
    "sw": {
        "greeting": "Habari ! Mimi ni DiaBot-RDC, mwenza wako wa kufuatilia sukari yako. U hali gani leo ?",
        "greeting_evening": "Habari ya jioni ! Siku yako ilikuwaje ? Umechukua vipimo vyako ?",
        "ask_measurement": "Umechukua kipimo cha sukari ya damu leo ? Usisahau, ni muhimu kwa afya yako !",
        "measurement_recorded": "✅ Sukari yako ya {valeur} g/L imehifadhiwa vizuri. {commentaire}",
        "measurement_normal": "🟢 Sukari yako iko katika kiwango cha kawaida. Endelea hivyo, unafanya vizuri !",
        "measurement_hypo_mild": "🟡 Sukari yako imeshuka kidogo ({valeur} g/L). Kula tunda au sukari kidogo na upime tena baada ya dakika 15.",
        "measurement_hypo_severe": "🔴 Tahadhari ! Sukari yako imeshuka sana ({valeur} g/L). Kunywa juisi au sukari haraka na mwite mtu karibu nawe. Kama haipandi, nenda hospitali.",
        "measurement_hyper_mild": "🟡 Sukari yako imepanda kidogo ({valeur} g/L). Kunywa maji mengi, tembea kidogo ikiwezekana, na angalia tena baada ya saa 2.",
        "measurement_hyper_severe": "🔴 Sukari yako imepanda sana ({valeur} g/L). Kunywa maji mengi, epuka vyakula vya sukari, na muone daktari haraka.",
        "measurement_urgence": "🚨 DHARURA ! Sukari yako ni {valeur} g/L. Hii ni hatari. Nenda HOSPITALI MARA MOJA au piga {numero_urgence}.",
        "cannot_diagnose": "Ninaelewa wasiwasi wako, lakini mimi si daktari na siwezi kukupa utambuzi. Tafadhali muone daktari.",
        "cannot_prescribe": "Siwezi kukuandikia dawa au kubadilisha matibabu yako. Daktari wako pekee ndiye anayeweza kufanya hivyo.",
        "goodbye": "Kwaheri na ujitunze vizuri ! 🌟",
    },
    "lu": {
        "greeting": "Mwapoleni ! Ami ni DiaBot-RDC, mukwenu wa kulonda sukadi yenu. Mudi bwino lelu ?",
        "greeting_evening": "Dilolo dibwino ! Dikuwa dia yenu diendejeje bua bua ? Wapimile bipimilu byenu ?",
        "ask_measurement": "Wapimile sukadi ya mashi lelu ? Kudimueka ku, kudi kwa mushinga mpo ku buumi bwenu !",
        "measurement_recorded": "✅ Sukadi yenu ya {valeur} g/L ibikidilwe bwino. {commentaire}",
        "measurement_normal": "🟢 Sukadi yenu idi mu tshilema tshibwino. Dikalejele bua bua, wudi kusala bwino !",
        "measurement_hypo_mild": "🟡 Sukadi yenu idikite panshi ({valeur} g/L). Dikudie tshibudi to sukadi nanku ne upime kabidi pa miniti 15.",
        "measurement_hypo_severe": "🔴 Dikeba ! Sukadi yenu idikite panshi mingi ({valeur} g/L). Dikudie sukadi nanku ne kenda ku lupitalu.",
        "measurement_urgence": "🚨 BWESHI ! Sukadi yenu idi pa {valeur} g/L. Kenda KU LUPITALU NANKU to ubike {numero_urgence}.",
        "cannot_diagnose": "Ndimanye mpasi yenu, kashi ami kadi muganga ku. Nkusungile kuya ku muganga.",
        "cannot_prescribe": "Nkudi kukuha mushipu ku. Muganga wenu nde udi ukusala nanku.",
        "goodbye": "Tushalane ne udikebe ne buumi bwenu ! 🌟",
    },
    "kg": {
        "greeting": "Mbote ! Mono i DiaBot-RDC, nduku na nge ya kulanda sukadi na nge. Nge kele mbote bubu ?",
        "greeting_evening": "Nkokila mbote ! Kilumbu na nge vandaka nki mutindu ? Nge me baka ntalu na nge ?",
        "ask_measurement": "Nge me baka ntalu ya sukadi ya menga bubu ? Kuvila ko, yo kele mfunu sambu na mavimpi na nge !",
        "measurement_recorded": "✅ Sukadi na nge ya {valeur} g/L me bumbama mbote. {commentaire}",
        "measurement_normal": "🟢 Sukadi na nge kele na kisika ya mbote. Landila mutindu yina, nge ke sala mbote !",
        "measurement_hypo_mild": "🟡 Sukadi na nge me kuma fioti ({valeur} g/L). Dia mbuma to sukadi fioti mpi baka ntalu diaka na nima ya miniti 15.",
        "measurement_hypo_severe": "🔴 Keba ! Sukadi na nge me kuma mingi na nsi ({valeur} g/L). Nwa sukadi nswalu mpi kwenda na lupitalu.",
        "measurement_urgence": "🚨 MPASI ! Sukadi na nge kele na {valeur} g/L. Kwenda KU LUPITALU NTANGU YINA to binga {numero_urgence}.",
        "cannot_diagnose": "Mono me wa mpasi na nge, kansi mono kele nganga ko. Kwenda na nganga na nge.",
        "cannot_prescribe": "Mono lenda pesa nkisi ko. Nganga na nge nde lenda sala yo.",
        "goodbye": "Tikala mbote mpi keba nitu na nge ! 🌟",
    },
}

URGENCE_MESSAGES: dict[str, dict[str, str]] = {
    "fr": {
        "hypo_critique": "🚨 URGENCE MÉDICALE : Glycémie très basse ({valeur} g/L) ! Prenez immédiatement du sucre (jus, miel, bonbon). Si la personne est inconsciente, NE DONNEZ RIEN PAR LA BOUCHE. Appelez le 112 ou allez à l'hôpital IMMÉDIATEMENT.",
        "hyper_critique": "🚨 URGENCE MÉDICALE : Glycémie très élevée ({valeur} g/L) ! Risque d'acidocétose. Buvez de l'eau en abondance. Rendez-vous IMMÉDIATEMENT aux urgences de l'hôpital le plus proche.",
        "appel_urgence": "Appelez le 112 (Police/SAMU) ou rendez-vous à l'hôpital le plus proche immédiatement.",
    },
    "ln": {
        "hypo_critique": "🚨 LIKAMA YA MONGANGA : Sukadi ekiti mingi ({valeur} g/L) ! Mela sukadi nokinoki (jus, nzoi, bonbon). Kende lopitalo MBALA MOKO.",
        "hyper_critique": "🚨 LIKAMA YA MONGANGA : Sukadi emati mingi ({valeur} g/L) ! Mela mai mingi. Kende lopitalo MBALA MOKO.",
        "appel_urgence": "Bengá 112 to kende lopitalo oyo ezali pene mbala moko.",
    },
    "sw": {
        "hypo_critique": "🚨 DHARURA YA KIMATIBABU : Sukari imeshuka sana ({valeur} g/L) ! Kunywa sukari haraka. Nenda hospitali MARA MOJA.",
        "hyper_critique": "🚨 DHARURA YA KIMATIBABU : Sukari imepanda sana ({valeur} g/L) ! Kunywa maji mengi. Nenda hospitali MARA MOJA.",
        "appel_urgence": "Piga 112 au nenda hospitali ya karibu mara moja.",
    },
    "lu": {
        "hypo_critique": "🚨 BWESHI BWA MUGANGA : Sukadi idikite panshi mingi ({valeur} g/L) ! Kudie sukadi nanku. Kenda ku lupitalu NANKU.",
        "hyper_critique": "🚨 BWESHI BWA MUGANGA : Sukadi idimuke mingi ({valeur} g/L) ! Kunwa mai mingi. Kenda ku lupitalu NANKU.",
        "appel_urgence": "Bike 112 to kenda ku lupitalu lupiipi nanku.",
    },
    "kg": {
        "hypo_critique": "🚨 MPASI YA NGANGA : Sukadi me kuma na nsi mingi ({valeur} g/L) ! Nwa sukadi nswalu. Kwenda na lupitalu NTANGU YINA.",
        "hyper_critique": "🚨 MPASI YA NGANGA : Sukadi me kuma na zulu mingi ({valeur} g/L) ! Nwa masa mingi. Kwenda na lupitalu NTANGU YINA.",
        "appel_urgence": "Binga 112 to kwenda na lupitalu ya penele ntangu yina.",
    },
}

NUTRITION_RDC: dict[str, dict[str, dict[str, Any]]] = {
    "fr": {
        "fufu_manioc": {"nom": "Fufu de manioc (kwanga)", "ig": "moyen", "conseil": "Préférez-le au fufu de maïs. Portion : taille d'un poing.", "charge_glycemique": 25},
        "fufu_mais": {"nom": "Fufu de maïs", "ig": "élevé", "conseil": "Limitez la portion. Accompagnez de beaucoup de légumes (pondu).", "charge_glycemique": 35},
        "pondu": {"nom": "Pondu (feuilles de manioc)", "ig": "bas", "conseil": "Excellent choix ! Riche en fibres, il ralentit l'absorption du sucre.", "charge_glycemique": 5},
        "saka_saka": {"nom": "Saka-saka", "ig": "bas", "conseil": "Très bon pour la glycémie. Mangez-en généreusement.", "charge_glycemique": 5},
        "lenga_lenga": {"nom": "Lenga-lenga (amarante)", "ig": "bas", "conseil": "Légume-feuille excellent. Ajoutez-le à vos repas.", "charge_glycemique": 4},
        "makemba": {"nom": "Makemba (bananes plantain)", "ig": "moyen-élevé", "conseil": "Préférez les plantains verts (bouillis) aux mûrs (frits). Portion modérée.", "charge_glycemique": 20},
        "chikwangue": {"nom": "Chikwangue (bâton de manioc)", "ig": "moyen", "conseil": "Alternative au fufu. Portion modérée.", "charge_glycemique": 22},
        "riz": {"nom": "Riz blanc", "ig": "élevé", "conseil": "Limitez la quantité. Préférez le riz complet si disponible.", "charge_glycemique": 30},
        "haricots": {"nom": "Haricots (madesu)", "ig": "bas", "conseil": "Excellent ! Les légumineuses stabilisent la glycémie.", "charge_glycemique": 10},
        "poisson": {"nom": "Poisson (frais ou salé)", "ig": "nul", "conseil": "Très bon choix protéiné.", "charge_glycemique": 0},
        "viande": {"nom": "Viande (ngulu, ntaba, ngombe)", "ig": "nul", "conseil": "Bonne source de protéines. Évitez les fritures excessives.", "charge_glycemique": 0},
        "mikate": {"nom": "Mikate (beignets)", "ig": "très élevé", "conseil": "⚠️ À éviter autant que possible. Très riche en sucre et en graisse.", "charge_glycemique": 40},
        "avocat": {"nom": "Avocat", "ig": "bas", "conseil": "Excellent ! Gras sain qui ralentit l'absorption du sucre.", "charge_glycemique": 2},
        "arachides": {"nom": "Arachides (nguba)", "ig": "bas", "conseil": "Bon en petite quantité. Riche en protéines et bon gras.", "charge_glycemique": 3},
    },
}
for _l in ("ln", "sw", "lu", "kg"):
    NUTRITION_RDC[_l] = NUTRITION_RDC["fr"]


# ============================================================================
# DÉTECTION DE LANGUE (MARKERS)
# ============================================================================

LANGUE_MARKERS: dict[str, list[str]] = {
    "fr": [
        "le", "la", "les", "de", "du", "des", "un", "une", "et", "est",
        "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
        "dans", "pour", "avec", "sur", "pas", "que", "qui", "mais",
        "glycémie", "médecin", "diabète", "sang", "sucre", "bonjour",
    ],
    "ln": [
        "na", "ya", "ezali", "nazali", "ozali", "azali", "tozali",
        "mpe", "kasi", "te", "mbote", "ndenge", "nini", "ngai", "yo", "ye",
        "makila", "sukadi", "monganga", "lopitalo", "kolia", "kolala",
    ],
    "sw": [
        "na", "ya", "wa", "ni", "kwa", "kuwa", "hii", "mimi", "wewe", "yeye",
        "habari", "asante", "tafadhali", "kwaheri", "ndiyo", "hapana",
        "sukari", "damu", "daktari", "hospitali", "dawa", "mwili",
    ],
    "lu": [
        "ne", "wa", "bu", "udi", "wudi", "mudi", "badi", "tudi",
        "meme", "ewe", "yeye", "sukadi", "mashi", "muganga", "lupitalu",
        "mushipu", "mubidi", "kudia", "kunwa", "kulala", "lelu",
    ],
    "kg": [
        "na", "ya", "ke", "me", "ve", "mono", "nge", "yandi",
        "mbote", "matondo", "sambu", "sukadi", "menga", "nganga",
        "lupitalu", "nkisi", "nitu", "kudia", "kunwa", "kulala", "bubu",
    ],
}


class LanguageDetector:
    """Détecteur automatique de langue."""

    def __init__(self) -> None:
        self._markers = LANGUE_MARKERS

    def detect(self, texte: str) -> TraductionResultat:
        if not texte or not texte.strip():
            return TraductionResultat(
                texte=texte,
                langue_detectee=LANGUE_PAR_DEFAUT.value,
                confiance=0.0,
            )

        mots = re.findall(r"\b\w+\b", texte.lower())
        if not mots:
            return TraductionResultat(
                texte=texte,
                langue_detectee=LANGUE_PAR_DEFAUT.value,
                confiance=0.0,
            )

        scores: dict[str, float] = {}
        for langue, markers in self._markers.items():
            marker_set = set(markers)
            matches = sum(1 for m in mots if m in marker_set)
            scores[langue] = matches / len(mots) if mots else 0.0

        sorted_langs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_lang = sorted_langs[0][0]
        best_score = sorted_langs[0][1]

        if best_score < 0.05:
            best_lang = LANGUE_PAR_DEFAUT.value

        return TraductionResultat(
            texte=texte,
            langue_detectee=best_lang,
            confiance=round(min(best_score * 3, 1.0), 2),
        )


# ============================================================================
# MOTEUR MULTILINGUE
# ============================================================================

class MultilingualEngine:
    """Moteur de traduction et localisation."""

    def __init__(self, default_lang: str = LANGUE_PAR_DEFAUT.value) -> None:
        self.default_lang = default_lang
        self.detector = LanguageDetector()
        self._dictionaries: dict[str, dict[str, dict[str, str]]] = {
            "ui": UI_TRADUCTIONS,
            "ia": IA_MESSAGES,
            "urgence": URGENCE_MESSAGES,
        }

    def translate(self, key: str, langue: str, category: str = "ui", **kwargs: Any) -> str:
        cat_dict = self._dictionaries.get(category, {})
        lang_dict = cat_dict.get(langue, {})
        texte = lang_dict.get(key)

        if texte is None and langue != self.default_lang:
            texte = cat_dict.get(self.default_lang, {}).get(key)

        if texte is None:
            return f"[{key}]"

        try:
            return texte.format(**kwargs) if kwargs else texte
        except Exception:
            return texte

    def translate_ia_message(self, key: str, langue: str, **kwargs: Any) -> str:
        return self.translate(key, langue, category="ia", **kwargs)

    def translate_urgence(self, key: str, langue: str, **kwargs: Any) -> str:
        return self.translate(key, langue, category="urgence", **kwargs)

    def detect_language(self, texte: str) -> TraductionResultat:
        return self.detector.detect(texte)

    def get_nutrition_conseil(self, aliment: str, langue: str) -> Optional[dict[str, Any]]:
        lang_dict = NUTRITION_RDC.get(langue, NUTRITION_RDC["fr"])
        return lang_dict.get(aliment)

    def get_supported_languages(self) -> list[dict[str, str]]:
        return [
            {
                "code": lang.value,
                "nom": LANGUES_INFO[lang.value]["nom"],
                "nom_natif": LANGUES_INFO[lang.value]["nom_natif"],
            }
            for lang in LangueSupportee
        ]


# ============================================================================
# INSTANCE GLOBALE
# ============================================================================

multilingual_engine = MultilingualEngine()

__all__ = [
    "MultilingualEngine", "LanguageDetector", "multilingual_engine",
    "TraductionResultat", "MessageLocalise",
]