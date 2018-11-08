function init() {
    var canvas = document.getElementById("the_canvas");
    var ctx = canvas.getContext("2d");
    canvas.width = 240;
    canvas.height = 240;
}
window.onload = init();

document.getElementById('generate-button').onclick = ui_generate_button_event_listener;

function ui_generate_button_event_listener(event) {
    tf.loadModel('model/model.json').then(handleModel).catch(handleError);
    function handleModel(model) {
        const y = tf.tidy(() => {
            const z = tf.randomNormal([1, 100]);
            // const noise = tf.expandDims(z[0], axis = 0)
            const y = model.predict(z).squeeze().div(tf.scalar(2)).add(tf.scalar(0.5));
            return image_enlarge(y, 4);
        });
        let c = document.getElementById("the_canvas");
        tf.toPixels(y, c);
    }
    function handleError(error) {
        console.log("model error")
    }

}

function image_enlarge(y, draw_multiplier) {
    if (draw_multiplier === 1) {
        return y;
    }
    let size = y.shape[0];
    return y.expandDims(2).tile([1, 1, draw_multiplier, 1]
    ).reshape([size, size * draw_multiplier, 3]
    ).expandDims(1).tile([1, draw_multiplier, 1, 1]
    ).reshape([size * draw_multiplier, size * draw_multiplier, 3])
}
