"""Class-level symptom descriptions (SET1 in the paper), written offline.

No target (PlantDoc) data was used to write or select them.
"""

SET1 = {
    0: ["a photograph of a healthy apple leaf, uniformly green with a smooth glossy surface and no discoloration",
        "an apple tree leaf in good health, oval with finely serrated edges and clear vein structure",
        "a green apple leaf free of any spots, lesions or yellowing",
        "close-up of an unblemished apple leaf showing even green pigmentation throughout"],
    1: ["an apple leaf infected with cedar apple rust, bright orange and yellow circular spots on the upper surface",
        "apple rust disease on a leaf, vivid orange-yellow lesions ringed by a darker border",
        "a leaf with rust fungus, raised orange pustules scattered across the green surface",
        "cedar apple rust symptoms, small yellow spots enlarging into orange lesions with concentric rings"],
    2: ["a bell pepper leaf with bacterial spot disease, covered in numerous small dark brown water-soaked lesions",
        "pepper leaf showing bacterial leaf spot, angular black-brown spots with yellow halos",
        "a pepper plant leaf damaged by bacterial infection, many tiny necrotic spots and some yellowing",
        "close-up of bacterial spot on pepper foliage, small circular dark lesions across the blade"],
    3: ["a healthy bell pepper leaf, deep green with a smooth waxy surface and no blemishes",
        "an unblemished pepper plant leaf, broad and glossy with even coloration",
        "a green pepper leaf in perfect condition, free of spots or lesions",
        "close-up of a healthy capsicum leaf showing uniform green pigment"],
    4: ["a grape leaf with black rot disease, tan-brown circular lesions with dark purple-black margins",
        "black rot infection on a grape leaf, reddish-brown spots developing dark borders and tiny black specks",
        "grapevine leaf displaying black rot lesions, irregular brown patches surrounded by darker rings",
        "close-up of black rot on grape foliage, necrotic tan centres ringed in dark brown"],
    5: ["a healthy grape leaf, bright green with characteristic lobed shape and prominent veins",
        "an unblemished grapevine leaf, deeply lobed with serrated margins and even green colour",
        "a green grape leaf free of any disease spots or discoloration",
        "close-up of a healthy grapevine leaf showing clean surface and clear venation"],
    6: ["a healthy tomato leaf, medium green with deeply serrated compound leaflets and no spots",
        "an unblemished tomato plant leaf showing even green colour and slightly fuzzy texture",
        "a tomato leaf in good condition, free of lesions, yellowing or spotting",
        "close-up of healthy tomato foliage with clean green leaflets"],
    7: ["a tomato leaf with septoria leaf spot, many small circular spots with grey-white centres and dark brown margins",
        "septoria infection on tomato foliage, numerous tiny round lesions with pale centres scattered across the leaflet",
        "tomato leaf showing septoria disease, small water-soaked spots turning grey with dark edges and surrounding yellowing",
        "close-up of septoria leaf spot symptoms, many small circular tan lesions with dark borders on a tomato leaflet"],
}

# Plain class names, used by the CLIP prompting baselines and CoOp
PROMPT_NAMES = [
    "apple leaf (healthy)", "apple leaf with cedar apple rust", "bell pepper leaf with bacterial spot",
    "bell pepper leaf (healthy)", "grape leaf with black rot", "grape leaf (healthy)",
    "tomato leaf (healthy)", "tomato leaf with septoria leaf spot",
]
