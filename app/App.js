import React, { useState, useEffect } from "react";
import { Button, Space, Upload, message } from "antd";
import { UploadOutlined } from "@ant-design/icons";
import * as tf from "@tensorflow/tfjs";
import { file2img, img2x } from "./utils";

const DATA_URL = "http://127.0.0.1:8080/";

const App = () => {
	let modal = null;
    let CLASSES = null;
	const props = {
		name: "file",
		async onChange(info) {
			if (info.file.status !== "uploading") {
				const file = await file2img(info.file.originFileObj);
                const pred = tf.tidy(() => {
                    const x = img2x(file);
                    return modal.predict(x);
                });
                const results = pred.arraySync()[0].map((score, i) => {
                    return {
                        score, 
                        label: CLASSES[i]
                    }
                }).sort((a, b) => b.score - a.score);
                console.log(results);
			}
		},
	};
	useEffect(() => {
		const fetchData = async () => {
			modal = await tf.loadLayersModel(DATA_URL + "model.json");
            CLASSES = await fetch(DATA_URL + 'classes.json').then(res => res.json());
			// modal.summary();
		};
		fetchData().catch((error) => console.error(error));
        
	}, []);

	return (
		<Space wrap>
			<Upload {...props}>
				<Button icon={<UploadOutlined />}>Click to Upload</Button>
			</Upload>
		</Space>
	);
};
export default App;
