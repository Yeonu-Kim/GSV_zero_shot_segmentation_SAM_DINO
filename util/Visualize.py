import matplotlib.pyplot as plt

def show_cubemap(cubemap_faces):
    plt.figure(figsize=(16, 8))
    for i, (face, img) in enumerate(cubemap_faces.items()):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(face)
        plt.axis('off')

    plt.tight_layout()
    plt.show()