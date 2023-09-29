fn main():
       from python import Python

    # test using PIL in Mojo

    let pil = Python.import("PIL.Image")
    let img = pil.open("test.png")
    img.show()

    