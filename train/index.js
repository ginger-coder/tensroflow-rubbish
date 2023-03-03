const tf = require('@tensorflow/tfjs-node');
const getData = require('./data');

const TRAIN_DIR = '垃圾分类';
const OUTPUT_DIR = 'output';
const MOBILE_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json';
const main = async () => {
    // 加载数据
    const { xs, ys, classes } = await getData(TRAIN_DIR, OUTPUT_DIR);
    // 定义模型 截断模型 + 双层神经网络 （迁移训练）
    const mobilenet = await tf.loadLayersModel(MOBILE_URL);
    mobilenet.summary();

    const model = tf.sequential();
    for (let i = 0; i <= 86; i++) {
       const layer = mobilenet.layers[i];
       layer.trainable = false;
       model.add(layer);
    }
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({
        units: 10,
        activation: 'relu',
    }))
    model.add(tf.layers.dense({
        units: classes.length,
        activation: 'softmax'
    }))
    // 训练模型 损失函数 优化器
};

main();