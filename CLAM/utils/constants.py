IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]
VAE_MEAN = [0.5, 0.5, 0.5]
VAE_STD = [0.5, 0.5, 0.5]

MODEL2CONSTANTS = {
	"resnet50_trunc": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "r18": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"uni_v1":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"conch_v1":
	{
		"mean": OPENAI_MEAN,
		"std": OPENAI_STD
	},
    "sd_vae":
	{
		"mean": VAE_MEAN,
		"std": VAE_STD
	},
	"chief":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"gigap":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
}