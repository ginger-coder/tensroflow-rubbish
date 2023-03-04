const fs = require("fs");
const tf = require("@tensorflow/tfjs-node-gpu");

const img2x = (imgPath) => {
	const buffer = fs.readFileSync(imgPath);
	return tf.tidy(() => {
		const imgTs = tf.node.decodeImage(new Uint8Array(buffer));
		// 要求224， 224
		const imgTsResiced = tf.image.resizeBilinear(imgTs, [224, 224]);
		// 像素 -1 到 1
		return imgTsResiced
			.toFloat()
			.sub(255 / 2)
			.div(255 / 2)
			.reshape([1, 224, 224, 3]);
	});
};
const getData = async (trainDir, outputDir) => {
	const classes = fs.readdirSync(trainDir).filter( n => !n.includes('DS_Store'));
	fs.writeFileSync(`${outputDir}/classes.json`, JSON.stringify(classes));

	const data = [];
	classes.forEach((dir, dirIndex) => {
		fs.readdirSync(`${trainDir}/${dir}`)
			.filter((n) => n.match(/(jpg|jpeg|png)$/))
			.slice(0, 2)
			.forEach((filename) => {
				const imgPath = `${trainDir}/${dir}/${filename}`;
				data.push({imgPath, dirIndex});
			});
	});

	tf.util.shuffle(data);

	const ds = tf.data.generator(function* () {
		const count = data.length;
		const batchSize = 1;
		for (let start = 0; start < count; start += batchSize) {
			const end = Math.min(start + batchSize, count);
			console.log('当前批次', start);
			yield tf.tidy(() => {
				const inputs = [];
				const labels = [];
				for (let j = 0; j < end; j += 1) {
					const {imgPath, dirIndex} = data[j];
					const x = img2x(imgPath);
					inputs.push(x);
					labels.push(dirIndex);
				}
				const xs = tf.concat(inputs);
    			const ys = tf.tensor(labels);
				return {
					xs,
					ys
				}
			});
		}
	})




    return {
		ds,
        classes
    }
};

module.exports = getData;
