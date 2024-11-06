##
# @file plot.py
# @author Keren Zhu
# @date Feb 2019
#

import db

def valid_image(sess, test_xs, test_ys, model, name):
    n_examples = len(test_xs)
    if (n_examples > 5):
        n_examples = 5
    test_xs_norm = np.array([img for img in test_xs])
    test_ys_norm = np.array([img for img in test_ys])
    #scale1(test_xs_norm)
    #scale1(test_ys_norm)
    recon = []
    fig, axs = plt.subplots(4, n_examples, figsize=(n_examples, 4))
    recon= sess.run(model['y'], feed_dict={model['x']: test_xs_norm, model['re']: test_ys_norm})
    cost = sess.run(model['cost'], feed_dict={model['x']: test_xs_norm, model['re']: test_ys_norm})
    print("Valid image", cost)
    for example_i in range(n_examples):
        xs_img = test_xs[example_i, :, :, 0]
        axs[0][example_i].imshow(
            xs_img)
        xs_img = test_xs[example_i, :, :, 1]
        axs[1][example_i].imshow(
            xs_img)
        ys_img = np.reshape(test_ys[example_i], [dimension, dimension])
        axs[2][example_i].imshow(
            ys_img)
        re_img = np.reshape(recon[example_i], [dimension, dimension])
        axs[3][example_i].imshow(
            re_img)
        """
        axs[2][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (dimension * dimension,)) ,
                (dimension, dimension)))
        """
    figname = "./fig/"+name + ".png"
    plt.savefig(figname)
    return cost

def test_image(sess, test_xs, model, name):
    n_examples = len(test_xs)
    test_xs_norm = np.array([img for img in test_xs])
    #scale1(test_xs_norm)
    #scale1(test_ys_norm)
    recon = []
    fig, axs = plt.subplots(3, n_examples, figsize=(n_examples, 3))
    recon= sess.run(model['y'], feed_dict={model['x']: test_xs_norm})
    for example_i in range(n_examples):
        xs_img = test_xs[example_i, :, :, 0]
        axs[0][example_i].imshow(
            xs_img)
        xs_img = test_xs[example_i, :, :, 1]
        axs[1][example_i].imshow(
            xs_img)
        re_img = np.reshape(recon[example_i], [dimension, dimension])
        axs[2][example_i].imshow(
            re_img)
        realname = "./fig/test_output_"+name+"_"+str(example_i)+".txt"
        util.export_grayscale_image_text(re_img, realname)
    figname = "./fig/"+name + ".png"
    plt.savefig(figname)

