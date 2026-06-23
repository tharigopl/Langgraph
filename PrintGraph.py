from PIL import Image
import io
import matplotlib.pyplot as plt

def print_graph(app):
    try:
        image_bytes = app.get_graph().draw_mermaid_png()
    except Exception as e:
        print(f"[print_graph] Could not render graph image: {e}")
        print(app.get_graph().draw_mermaid())   # fallback text
        return                                   # exit early, no image to show

    # only runs if image_bytes was fetched successfully
    image = Image.open(io.BytesIO(image_bytes))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def print_stream_values(stream):
    printed = 0
    for s in stream:
        messages = s["messages"]
        for message in messages[printed:]:
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
        printed = len(messages)