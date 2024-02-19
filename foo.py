if 1:
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.datasets import mnist

    TF_ENABLE_ONEDNN_OPTS=0

    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # print(train_images.shape)
    # print(train_images)
    # print(train_images[0])

    # print(train_labels.shape)
    # print(train_labels)
    print(train_images[59999])

    print(train_labels[59997])
    print(train_labels[59998])
    print(train_labels[59999])

    print(test_images[9999])

    print(test_labels[9997])
    print(test_labels[9998])
    print(test_labels[9999])


    # plt.imshow(train_images[0], cmap='gray')
    # plt.show()
    # plt.imshow(train_images[1], cmap='gray')
    # plt.show()
    # plt.imshow(train_images[2], cmap='gray')
    # plt.show()

    # plt.imshow(train_labels[0], cmap='gray')
    # plt.imshow(train_labels[1], cmap='gray')

    # Step 1: Input Layer
    # Use the first image in the dataset as an example
    input_image = train_images[0]

    # Display the input image
    plt.imshow(input_image, cmap='gray')
    plt.title('Input Image')
    plt.show()

    # Step 2: Convolutional Layer (Conv2D)
    # Assume we have a 3x3 learnable filter for illustration
    filter_weights = np.array([[1, -1, 1],
                            [0,  1, 0],
                            [-1, 0, 1]])

    # Apply 2D convolution manually
    def conv2d(input_data, filter_weights):
        filter_height, filter_width = filter_weights.shape
        input_height, input_width = input_data.shape

        output_height = input_height - filter_height + 1
        output_width = input_width - filter_width + 1

        output = np.zeros((output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                region = input_data[i:i + filter_height, j:j + filter_width]
                output[i, j] = np.sum(region * filter_weights)

        return output

    # Apply convolution to the input image
    convolution_result = conv2d(input_image, filter_weights)

    # Display the result of the convolution
    plt.imshow(convolution_result, cmap='gray')
    plt.title('Convolution Result')
    plt.show()


# Example of displaying MNIST dataset using matplotlib
if 0:
    import matplotlib.pyplot as plt
    from tensorflow.keras.datasets import mnist  # Assuming you have TensorFlow installed

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(x_train[0])
    print(x_train[0].shape)
    print(x_train.shape)

    # Display a few images
    num_images_to_display = 5

    for i in range(num_images_to_display):
        plt.subplot(1, num_images_to_display, i + 1)
        plt.imshow(x_train[i], cmap='gray')
        plt.title(f"Label: {y_train[i]}")
        plt.axis('off')

    plt.show()