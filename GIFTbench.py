import sys
import os
sys.path.append(os.path.abspath('.'))
import gradio as gr

from mnist.mnist_gradiovae import run_vae_tig as run_mnist_vae
from mnist.mnist_gradiogan import run_gan_tig as run_mnist_gan
from mnist.mnist_gradiodm import run_diffusion_tig as run_mnist_dm

from svhn.svhn_gradiovae import run_vae_tig1 as run_svhn_vae
from svhn.svhn_gradiogan import run_gan_tig1 as run_svhn_gan
from svhn.svhn_gradiodm import run_diffusion_tig1 as run_svhn_dm

from cifar10.cifar10_gradiovae import run_vae_tig2 as run_cifar10_vae
from cifar10.cifar10_gradiogan import run_gan_tig2 as run_cifar10_gan
from cifar10.cifar10_gradiodm import run_diffusion_tig2 as run_cifar10_dm

from ImageNet.imagenet_gradiovae_pizza import run_vae_tig_pizza as run_imagenet_vae
from ImageNet.imagenet_gradiobiggan_pizza import run_biggan_tig_pizza as run_imagenet_biggan
from ImageNet.imagenet_gradiodm_pizza import run_diffusion_tig_pizza as run_imagenet_dm

from ImageNet.imagenet_gradiovae_teddy import run_vae_tig_teddy as run_imagenet_vae1
from ImageNet.imagenet_gradiobiggan_teddy import run_biggan_tig_teddy as run_imagenet_biggan1
from ImageNet.imagenet_gradiodm_teddy import run_diffusion_tig_teddy as run_imagenet_dm1

# Prompts
mnist_prompts = [
    "A photo of Z0ero Number0", "A photo of one1 Number1", "A photo of two2 Number2",
    "A photo of three3 Number3", "A photo of Four4 Number4", "A photo of Five5 Number5",
    "A photo of Six6 Number6", "A photo of Seven7 Number7", "A photo of Eight8 Number8",
    "A photo of Nine9 Number9"
]

svhn_prompts = [
    "A photo of HouseNo0 Hnumber0", "A photo of HouseNo1 Hnumber1", "A photo of HouseNo2 Hnumber2",
    "A photo of HouseNo3 Hnumber3", "A photo of HouseNo4 Hnumber4", "A photo of HouseNo5 Hnumber5",
    "A photo of HouseNo6 Hnumber6", "A photo of HouseNo7 Hnumber7", "A photo of HouseNo8 Hnumber8",
    "A photo of HouseNo9 Hnumber9"
]

cifar10_prompts = [
    "A photo of A1plane0 cifar10_0", "A photo of car1 cifar10_1", "A photo of bird2 cifar10_2",
    "A photo of cat3 cifar10_3", "A photo of deer4 cifar10_4", "A photo of dog5 cifar10_5",
    "A photo of frog6 cifar10_6", "A photo of horse7 cifar10_7", "A photo of ship8 cifar10_8",
    "A photo of truck9 cifar10_9"
]

imagenet_prompt = ["A photo of 1pizza pizza_slice"]
imagenet1_prompt = ["A photo of 1toy_bear teddy_bear"]
# Function to toggle visibility of the upload field
def toggle_upload(classifier_choice):
    return gr.update(visible=(classifier_choice == "Upload Custom"))

def validate_classifier_file(classifier_file):
    if classifier_file is None:
        return "No file uploaded."

    if not classifier_file.name.endswith(".jit"):
        return "Error: Only .jit TorchScript files are supported."

    try:
        classifier = torch.jit.load(classifier_file.name, map_location=device)
        classifier.eval()
        return "Custom TorchScript classifier loaded successfully."
    except Exception as e:
        return f"Error loading classifier: {e}"


# Dynamic parameter updater
def update_params_mnist(model_choice):
    if model_choice == "VAE":
        return gr.update(choices=[
                ("Low", 0.00073212788105011),
                ("High", 0.0073212788105011)
            ], value=0.00073212788105011), gr.update(choices=[
                ("Low", 0.00146425576210022),
                ("High", 0.0146425576210022)
            ], value=0.00146425576210022), gr.update(visible=False)
    elif model_choice == "GAN":
        return gr.update(choices=[
                ("Low", 0.000399987590312958),
                ("High", 0.00399987590312958)
            ], value=0.000399987590312958), gr.update(choices=[
                ("Low", 0.000799975180625916),
                ("High", 0.00799975180625916)
            ], value=0.000799975180625916), gr.update(visible=False)
    elif model_choice == "DM":
        return gr.update(choices=[
                ("Low", 0.00108344359397888),
                ("High", 0.0108344359397888)
            ], value=0.00108344359397888), gr.update(choices=[
                ("Low", 0.00216688718795776),
                ("High", 0.0216688718795776)
            ], value=0.00216688718795776), gr.update(choices=mnist_prompts, visible=True)

def update_params_svhn(model_choice):
    if model_choice == "VAE":
        return gr.update(choices=[
                ("Low", 0.000999399948120117),
                ("High", 0.00999399948120117)
            ], value=0.000999399948120117), gr.update(choices=[
                ("Low", 0.00199879989624023),
                ("High", 0.0199879989624023)
            ], value=0.00199879989624023), gr.update(visible=False)
    elif model_choice == "GAN":
        return gr.update(choices=[
                ("Low", 0.000862655591964722),
                ("High", 0.00862655591964722)
            ], value=0.000862655591964722), gr.update(choices=[
                ("Low", 0.00172531118392944),
                ("High", 0.0172531118392944)
            ], value=0.00172531118392944), gr.update(visible=False)
    elif model_choice == "DM":
        return gr.update(choices=[
                ("Low", 0.00108913173675537),
                ("High", 0.0108913173675537)
            ], value=0.00108913173675537), gr.update(choices=[
                ("Low", 0.00217826347351074),
                ("High", 0.0217826347351074)
            ], value=0.00217826347351074), gr.update(choices=svhn_prompts, visible=True)

def update_params_cifar10(model_choice):
    if model_choice == "VAE":
        return gr.update(choices=[
                ("Low", 0.000801093220710754),
                ("High", 0.00801093220710754)
            ],value =0.000801093220710754), gr.update(choices=[
                ("Low", 0.00160218644142151),
                ("High", 0.0160218644142151)
            ], value =0.00160218644142151), gr.update(visible=False)
    elif model_choice == "GAN":
        return gr.update(choices=[
                ("Low", 0.000910532093048096),
               ("High",0.00910532093048096)
            ], value= 0.000910532093048096), gr.update(choices=[
                ("Low", 0.00182106418609619),
                ("High",0.0182106418609619)
            ], value=0.00182106418609619), gr.update(visible=False)
    elif model_choice == "DM":
        return gr.update(choices=[
                ("Low", 0.00109171276092529),
                ("High", 0.0109171276092529)
            ], value=0.00109171276092529), gr.update(choices=[
                ("Low", 0.00218342552185059),
                ("High", 0.0218342552185059)
            ], value= 0.00218342552185059), gr.update(choices=cifar10_prompts, visible=True)

def update_params_imagenet(model_choice):
    if model_choice == "VAE (pizza)":
        return gr.update(choices=[
                ("Low",0.00172651433944702),
                ("High", 0.0172651433944702)
            ], value= 0.00172651433944702), gr.update(choices=[
                ("Low",0.00345302867889404 ),
                ("High",0.0345302867889404)
            ], value= 0.00345302867889404), gr.update(visible=False), gr.update(visible=False)
    elif model_choice == "GAN (pizza)":
        return gr.update(choices=[
                ("Low",0.000399987590312958),
                ("High",0.00399987590312958)
            ], value=0.000399987590312958), gr.update(choices=[
                ("Low",0.000799975180625916 ),
                ("High",0.00799975180625916)
            ], value=0.000799975180625916), gr.update(visible=False), gr.update(value = 1.0, visible=True)
    elif model_choice == "DM (pizza)":
        return gr.update(choices=[
                ("Low",0.00108863949775696 ),
                ("High",0.0108863949775696)
            ], value=0.00108863949775696), gr.update(choices=[
                ("Low",0.00217727899551392 ),
                ("High",0.0217727899551392)
            ], value=0.00217727899551392), gr.update(choices=imagenet_prompt, visible=True), gr.update(visible=False)

def update_params_imagenet2(model_choice):
    if model_choice == "VAE (teddy)":
        return gr.update(choices=[
                ("Low",0.00172651433944702),
                ("High", 0.0172651433944702)
            ], value= 0.00172651433944702), gr.update(choices=[
                ("Low",0.00345302867889404 ),
                ("High",0.0345302867889404)
            ], value= 0.00345302867889404), gr.update(visible=False), gr.update(visible=False)
    elif model_choice == "GAN (teddy)":
        return gr.update(choices=[
                ("Low",0.000399987590312958),
                ("High",0.00399987590312958)
            ], value=0.000399987590312958), gr.update(choices=[
                ("Low",0.000799975180625916 ),
                ("High",0.00799975180625916)
            ], value=0.000799975180625916), gr.update(visible=False), gr.update(value = 1.0, visible=True)
    elif model_choice == "DM (teddy)":
        return gr.update(choices=[
                ("Low",0.00108863949775696 ),
                ("High",0.0108863949775696)
            ], value=0.00108863949775696), gr.update(choices=[
                ("Low",0.00217727899551392 ),
                ("High",0.0217727899551392)
            ], value=0.00217727899551392), gr.update(choices=imagenet1_prompt, visible=True), gr.update(visible=False)

# Routing functions
def run_mnist(model_choice, gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier, prompt,classifier_file):
    if model_choice == "VAE":
        yield from run_mnist_vae(gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier,classifier_file)
    elif model_choice == "GAN":
        yield from run_mnist_gan(gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier,classifier_file)
    elif model_choice == "DM":
        yield from run_mnist_dm(gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier, prompt,classifier_file)

def run_svhn(model_choice, gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier, prompt,classifier_file):
    if model_choice == "VAE":
        yield from run_svhn_vae(gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier,classifier_file)
    elif model_choice == "GAN":
        yield from run_svhn_gan(gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier,classifier_file)
    elif model_choice == "DM":
        yield from run_svhn_dm(gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier, prompt,classifier_file)

def run_cifar10(model_choice, gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier, prompt,classifier_file):
    if model_choice == "VAE":
        yield from run_cifar10_vae(gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier,classifier_file)
    elif model_choice == "GAN":
        yield from run_cifar10_gan(gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier,classifier_file)
    elif model_choice == "DM":
        yield from run_cifar10_dm(gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier, prompt,classifier_file)

def run_imagenet(model_choice, gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier, prompt, truncation,classifier_file):
    if model_choice == "VAE (pizza)":
        yield from run_imagenet_vae(gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier,classifier_file)
    elif model_choice == "GAN (pizza)":
        yield from run_imagenet_biggan(imgs_to_samp,gen_num, pop_size, best_left, perturb_size, initial_perturb_size, classifier, truncation,classifier_file)
    elif model_choice == "DM (pizza)":
        yield from run_imagenet_dm(gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier, prompt,classifier_file)


def run_imagenet2(model_choice, gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier, prompt, truncation,classifier_file):
    if model_choice == "VAE (teddy)":
        yield from run_imagenet_vae1(gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier,classifier_file)
    elif model_choice == "GAN (teddy)":
        yield from run_imagenet_biggan1(imgs_to_samp, gen_num, pop_size, best_left, perturb_size, initial_perturb_size, classifier, truncation,classifier_file)
    elif model_choice == "DM (teddy)":
        yield from run_imagenet_dm1(gen_num, pop_size, best_left, perturb_size, initial_perturb_size, imgs_to_samp, classifier, prompt,classifier_file)
# --- Interface ---

with gr.Blocks() as demo:
    gr.Markdown("## TEST INPUT GENERATORS – (VAE / GAN / Diffusion)")
 
 #with gr.Blocks(css=".orange-btn button {background-color: orange !important; color: white !important;}") as demo:
    with gr.Tab("MNIST"):
        model_mnist = gr.Radio(["VAE", "GAN", "DM"], label="Model Type")
        with gr.Row():
            gen_num_mnist = gr.Slider(100, 500, value=250,step=1, label="Generations")
            pop_size_mnist = gr.Slider(10, 50, value=25,step=1, label="Population Size")
            best_left_mnist = gr.Slider(5, 20, value=10,step=1, label="Best Left (Selection)")
            perturb_mnist = gr.Dropdown(label="Perturbation Size", choices=[])
            initial_perturb_mnist = gr.Dropdown(label="Initial Perturbation Size", choices=[])
            imgs_to_samp_mnist = gr.Slider(1, 100, value=3,step=1, label="Images to Sample")
            #image_size_mnist = gr.Dropdown([28, 32, 224, 512], value=28, label="Image Size (pixels)")
            classifier_dropdown = gr.Dropdown(choices=["deepconv", "Upload Custom"],value="deepconv",label="Classifier Under Test")
            classifier_upload = gr.File(label="Upload .jit TorchScript Classifier",type="filepath",file_types=[".jit"],visible=False)

           # prompt_mnist = gr.Dropdown(choices=[], label="Prompt (only for DM)", visible=False)
        with gr.Row():
            prompt_mnist = gr.Dropdown(choices=[], label="Prompt (only for DM)", visible=False)
            run_btn_mnist = gr.Button("Run MNIST TIG")

        with gr.Row():
            status_mnist = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            status_table = gr.Dataframe(headers=["# Image", "Expected Label", "Predicted Label", "# Iterations"], interactive=False)
            gallery_mnist = gr.Gallery(label="MNIST Images", columns=2, allow_preview=True)
            download_mnist = gr.File(label="Download Generated Images (.zip)")
        # Dynamic update for dropdowns
        classifier_dropdown.change(fn = toggle_upload,inputs=classifier_dropdown,outputs=classifier_upload)
        classifier_upload.change(fn=validate_classifier_file,inputs=classifier_upload, outputs=status_mnist)  
        model_mnist.change(update_params_mnist, inputs=model_mnist, outputs=[perturb_mnist, initial_perturb_mnist, prompt_mnist])


        run_btn_mnist.click(
            fn=run_mnist,
            inputs=[model_mnist, gen_num_mnist, pop_size_mnist, best_left_mnist,
                    perturb_mnist, initial_perturb_mnist, imgs_to_samp_mnist, classifier_dropdown, prompt_mnist,classifier_upload],
            outputs=[status_mnist, gallery_mnist, download_mnist, status_table]
        )
#demo.launch()
    with gr.Tab("SVHN"):
        model_svhn = gr.Radio(["VAE", "GAN", "DM"], label="Model Type", value="")
        with gr.Row():
            gen_num_svhn = gr.Slider(100, 500, value=250,step=1, label="Generations")
            pop_size_svhn = gr.Slider(10, 50, value=25,step=1, label="Population Size")
            best_left_svhn = gr.Slider(5, 20, value=10,step=1, label="Best Left (Selection)")
            perturb_svhn = gr.Dropdown(label="Perturbation Size", choices=[])
            initial_perturb_svhn = gr.Dropdown(label="Initial Perturbation Size", choices=[])
            imgs_to_samp_svhn = gr.Slider(1, 100, value=3, label="Images to Sample")
            #image_size_svhn = gr.Dropdown([28, 32, 224, 512], value=32, label="Image Size (pixels)")
            classifier_dropdown = gr.Dropdown(choices=["VGGNET", "Upload Custom"], value="VGGNET",label="Classifier Under Test")
           # prompt_svhn = gr.Dropdown(choices=[], label="Prompt (only for DM)", visible=False)
            classifier_upload = gr.File(label="Upload .jit TorchScript Classifier",type="filepath",file_types=[".jit"],visible=False)
        with gr.Row():
            run_btn_svhn = gr.Button("Run SVHN TIG")
            prompt_svhn = gr.Dropdown(choices=[], label="Prompt (only for DM)", visible=False)
        
        with gr.Row():
            status_svhn = gr.Textbox(label="Status", interactive=False)


        with gr.Row():

            status_table = gr.Dataframe(headers=["# Image", "Expected Label", "Predicted Label", "# Iterations"], interactive=False)

            gallery_svhn = gr.Gallery(label="SVHN Images", columns=2, allow_preview=True)
            download_svhn = gr.File(label="Download Generated Images (.zip)")

        # Dynamic update for dropdowns
        # Dynamic update for dropdowns
        classifier_dropdown.change(fn = toggle_upload,inputs=classifier_dropdown,outputs=classifier_upload)
        classifier_upload.change(fn=validate_classifier_file,inputs=classifier_upload, outputs=status_svhn)  
        model_svhn.change(update_params_svhn, inputs=model_svhn, outputs=[perturb_svhn, initial_perturb_svhn, prompt_svhn])

        run_btn_svhn.click(
            fn=run_svhn,
            inputs=[model_svhn, gen_num_svhn, pop_size_svhn, best_left_svhn,
                    perturb_svhn, initial_perturb_svhn, imgs_to_samp_svhn, classifier_dropdown, prompt_svhn,classifier_upload],
            outputs=[status_svhn, gallery_svhn, download_svhn, status_table]
        )
  
    with gr.Tab("CIFAR-10"):
        model_cifar10 = gr.Radio(["VAE", "GAN", "DM"], label="Model Type")
        with gr.Row():
            gen_num_cifar10 = gr.Slider(100, 500, value=250,step=1, label="Generations")
            pop_size_cifar10 = gr.Slider(10, 50, value=25,step=1, label="Population Size")
            best_left_cifar10 = gr.Slider(5, 20, value=10, step=1, label="Best Left (Selection)")
            perturb_cifar10 = gr.Dropdown(label="Perturbation Size", choices=[])
            initial_perturb_cifar10= gr.Dropdown(label="Initial Perturbation Size", choices=[])
            imgs_to_samp_cifar10 = gr.Slider(1, 100, value=3, step=1,label="Images to Sample")
           # image_size_cifar10 = gr.Dropdown([28, 32, 224, 512], value=32, label="Image Size (pixels)")
            classifier_dropdown = gr.Dropdown(choices=["VGGNET", "Upload Custom"], value="VGGNET",label="Classifier Under Test")
            classifier_upload = gr.File(label="Upload .jit TorchScript Classifier",type="filepath",file_types=[".jit"],visible=False)
           # prompt_cifar10 = gr.Dropdown(choices=[], label="Prompt (only for DM)", visible=False)
        with gr.Row():
            run_btn_cifar10 = gr.Button("Run CIFAR-10 TIG")
            prompt_cifar10 = gr.Dropdown(choices=[], label="Prompt (only for DM)", visible=False)
        with gr.Row():
            status_cifar10 = gr.Textbox(label="Status", interactive=False)
        
        with gr.Row():
            status_table = gr.Dataframe(headers=["# Image", "Expected Label", "Predicted Label", "# Iterations"], interactive=False)
            gallery_cifar10 = gr.Gallery(label="CIFAR-10 Images", columns=2, allow_preview=True)
            download_Cifar10 = gr.File(label="Download Generated Images (.zip)")
        # Dynamic update for dropdowns
        classifier_dropdown.change(fn = toggle_upload,inputs=classifier_dropdown,outputs=classifier_upload)
        classifier_upload.change(fn=validate_classifier_file,inputs=classifier_upload, outputs=status_cifar10)  
        model_cifar10.change(update_params_cifar10, inputs=model_cifar10, outputs=[perturb_cifar10, initial_perturb_cifar10, prompt_cifar10])

        run_btn_cifar10.click(
            fn=run_cifar10,
            inputs=[model_cifar10, gen_num_cifar10, pop_size_cifar10, best_left_cifar10,
                    perturb_cifar10, initial_perturb_cifar10, imgs_to_samp_cifar10, classifier_dropdown, prompt_cifar10,classifier_upload],
            outputs=[status_cifar10, gallery_cifar10,download_Cifar10, status_table]
        )

    with gr.Tab("ImageNet (class-pizza)"):
        model_imagenet = gr.Radio(["VAE (pizza)", "GAN (pizza)", "DM (pizza)"], label="Model Type", value="VAE")
        with gr.Row():
            gen_num_imagenet = gr.Slider(100, 500, value=250, step=1, label="Generations")
            pop_size_imagenet = gr.Slider(10, 50, value=25, step=1, label="Population Size")
            best_left_imagenet = gr.Slider(5, 20, value=10, step=1, label="Best Left (Selection)")
            perturb_imagenet = gr.Dropdown(label="Perturbation Size", choices=[])
            initial_perturb_imagenet = gr.Dropdown(label="Initial Perturbation Size", choices=[])
            imgs_to_samp_imagenet = gr.Slider(1, 100, value=3, step=1, label="Images to Sample")
           # image_size_imagenet = gr.Dropdown([28,32, 128, 224, 512], value=224, label="Image Size (pixels)")
            classifier_dropdown = gr.Dropdown(choices=["VGG19bn", "Upload Custom"], value="VGG19bn",label="Classifier Under Test")
            truncation_imagenet = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Truncation Value (BigGAN only)", visible=False)
           # prompt_imagenet = gr.Dropdown(choices=[], label="Prompt (only for DM)", visible=False)
            classifier_upload = gr.File(label="Upload .jit TorchScript Classifier",type="filepath",file_types=[".jit"],visible=False)
 
        with gr.Row():
            run_btn_imagenet = gr.Button("Run ImageNet (class-Pizza) TIG")
            prompt_imagenet = gr.Dropdown(choices=[], label="Prompt (only for DM)", visible=False)

        with gr.Row():

            status_imagenet = gr.Textbox(label="Status", interactive=False)
         
        with gr.Row():
            status_table = gr.Dataframe(headers=["# Image", "Expected Label", "Predicted Label", "# Iterations"], interactive=False)
            gallery_imagenet = gr.Gallery(label="ImageNet Images", columns=2, allow_preview=True)
            download_pizza = gr.File(label="Download Generated Images (.zip)")
        # Dynamic update for dropdowns
        classifier_dropdown.change(fn = toggle_upload,inputs=classifier_dropdown,outputs=classifier_upload)
        classifier_upload.change(fn=validate_classifier_file,inputs=classifier_upload, outputs=status_imagenet)  
        model_imagenet.change(update_params_imagenet, inputs=model_imagenet, outputs=[perturb_imagenet, initial_perturb_imagenet, prompt_imagenet,truncation_imagenet])

        run_btn_imagenet.click(
            fn=run_imagenet,
            inputs=[model_imagenet, gen_num_imagenet, pop_size_imagenet, best_left_imagenet,
                    perturb_imagenet, initial_perturb_imagenet, imgs_to_samp_imagenet, classifier_dropdown, prompt_imagenet, truncation_imagenet,classifier_upload],
            outputs=[status_imagenet, gallery_imagenet,download_pizza, status_table]
        )

    with gr.Tab("ImageNet (class-teddy)"):
        model_imagenet = gr.Radio(["VAE (teddy)", "GAN (teddy)", "DM (teddy)"], label="Model Type", value="VAE")
        with gr.Row():
            gen_num_imagenet = gr.Slider(100, 500, value=250, step=1, label="Generations")
            pop_size_imagenet = gr.Slider(10, 50, value=25, step=1, label="Population Size")
            best_left_imagenet = gr.Slider(5, 20, value=10, step=1, label="Best Left (Selection)")
            perturb_imagenet = gr.Dropdown(label="Perturbation Size", choices=[])
            initial_perturb_imagenet = gr.Dropdown(label="Initial Perturbation Size", choices=[])
            imgs_to_samp_imagenet = gr.Slider(1, 100, value=3, step=1, label="Images to Sample")
            #image_size_imagenet = gr.Dropdown([28, 32, 128, 224, 512], value=224, label="Image Size (pixels)")
            classifier_dropdown = gr.Dropdown(choices=["VGG19bn", "Upload Custom"], value="VGG19bn",label="Classifier Under Test")
            truncation_imagenet = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Truncation Value (BigGAN only)",visible=False)
           # prompt_imagenet = gr.Dropdown(choices=[], label="Prompt (only for DM)", visible=False)
            classifier_upload = gr.File(label="Upload .jit TorchScript Classifier",type="filepath",file_types=[".jit"],visible=False)
        with gr.Row():
            run_btn_imagenet = gr.Button("Run ImageNet (class-teddy) TIG", elem_classes="orange-btn")
            prompt_imagenet = gr.Dropdown(choices=[], label="Prompt (only for DM)", visible=False)

        with gr.Row():
            status_imagenet = gr.Textbox(label="Status")

        with gr.Row():
            status_table = gr.Dataframe(headers=["# Image", "Expected Label", "Predicted Label", "# Iterations"], interactive=False)
            gallery_imagenet = gr.Gallery(label="ImageNet Images", columns=2, allow_preview=True)
            download_teddy = gr.File(label="Download Generated Images (.zip)")
     
        # Dynamic update for dropdowns
        classifier_dropdown.change(fn = toggle_upload,inputs=classifier_dropdown,outputs=classifier_upload)
        classifier_upload.change(fn=validate_classifier_file,inputs=classifier_upload, outputs=status_imagenet)  

        model_imagenet.change(update_params_imagenet2, inputs=model_imagenet, outputs=[perturb_imagenet, initial_perturb_imagenet, prompt_imagenet,truncation_imagenet])

        run_btn_imagenet.click(
            fn=run_imagenet2,
            inputs=[model_imagenet, gen_num_imagenet, pop_size_imagenet, best_left_imagenet,
                    perturb_imagenet, initial_perturb_imagenet, imgs_to_samp_imagenet, classifier_dropdown, prompt_imagenet, truncation_imagenet,classifier_upload],
            outputs=[status_imagenet, gallery_imagenet,download_teddy, status_table]
        )

demo.launch(server_name="0.0.0.0", server_port=7860, share=True, prevent_thread_lock=False)

