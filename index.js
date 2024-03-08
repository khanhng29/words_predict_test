import { loadLayersModel, tensor1d, argMax } from '../test_project/node_modules/@tensorflow/tfjs/dist/tf.node.js';
import { io } from '../test_project/node_modules/@tensorflow/tfjs-node/dist/index.js';
import express from '../test_project/node_modules/express/index.js';
import * as tf from "@tensorflow/tfjs"
import mongoose from 'mongoose';
import bodyParser from 'body-parser';
import pkg1 from '../test_project/node_modules/fs-extra/lib/fs/index.js';


const { readFileSync } = pkg1;
const app = express();
app.use(express.json());
let model;
let word2index;
const MAX_LENGTH = 51;

mongoose.connect('mongodb://localhost:27017/testdb_5_3')
    .then(() => {
        console.log('mongoose connected');
    })
    .catch(() => {
        console.log('err');
    });
const newSchema = new mongoose.Schema({
    text: {
        type: String,
        required: true
    },
    predictedText: {
        type: String,
        required: true
    }
});

const TextModel = mongoose.model('predict_data', newSchema);

function tokenizer(text) {
    const cleanedText = text.replace(/[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n]/g, ' ');
    const words = cleanedText.toLowerCase().split(' ');
    return words;
}

function pad_token(tokenizedText, maxLength) {
    var paddedTokens = [];
    paddedTokens = tokenizedText.slice();
    while (paddedTokens.length < maxLength) {
        paddedTokens.unshift(0);
    }
    return paddedTokens;
}
const word2indexData = readFileSync('C:\\Users\\Admin\\Desktop\\test_project\\word_index2.json', 'utf-8');
const handler = io.fileSystem('C:\\Users\\Admin\\Desktop\\test_project\\model2\\model.json');
word2index = JSON.parse(word2indexData);
var index2word = Object.keys(word2index)

function getTokenisedWord(seedWord) {
    const _token = word2index[seedWord.toLowerCase()];
    return _token;

}
async function predict_function(text, num_generate) {
    try {
        // const input_text = text;
        // const word2indexData = readFileSync('C:\\Users\\Admin\\Desktop\\test_project\\word_index2.json', 'utf-8');
        // const handler = io.fileSystem('C:\\Users\\Admin\\Desktop\\test_project\\model2\\model.json');
        model = await tf.loadLayersModel(handler);
        let seedWordToken = []

        for (let i = 0; i < num_generate; i++) {
            const tokenizedText = tokenizer(text);
            seedWordToken.push(getTokenisedWord(tokenizedText[i]));
            const padded_token_text = (tf.tensor1d(pad_token(seedWordToken, MAX_LENGTH))).reshape([-1, MAX_LENGTH]);
            // console.log(padded_token_text.print())
            const predictions = await (model.predict(padded_token_text)).arraySync()[0];
            let max_val = predictions[0];
            for (const predict_val of predictions) {
                if (max_val < predict_val) {
                    max_val = predict_val;
                }
            }
            const resultIdx = predictions.indexOf(max_val);
            if (resultIdx === -1 || !index2word[resultIdx - 1]) {
                throw new Error('Predicted word not found in dictionary');
            }
            const word = index2word[resultIdx - 1];
            text = text + " " + word;
            console.log(text)
        }
        return text;

    } catch (error) {
        console.error(error);
        console.log("Failed to make prediction.")
        throw error; 
    }
}


app.use(bodyParser.urlencoded({ extended: true }));
app.post("/predict", async (req, res) => {
    try {
        const text = req.body.text;
        const numGenerate = req.body.num_words;
        
        if (!text || text.trim() === "") {
            throw new Error("Type smt ...");
        }

        let generatedText = await predict_function(text, numGenerate);
        res.send(generatedText);

        const newData = new TextModel({
            text: text,
            predictedText: generatedText
        });
        newData.save()
        console.log("saved")
    } catch (error) {
        console.error(error);
        res.status(500).send(error.message);
    }
});



app.get("/predict", async (req, res) => {
    try {
        const html = `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Words Recommend</title>
            </head>
            <center>
            <body>
                <h1>Words Recommend</h1>
                <form id="predict-form" action="/predict" method="POST">
                    <label for="text-area"></label><br>
                    <textarea placeholder="Type here ..." id="text-area" name="text" rows="4" cols="50"></textarea>
                    <br>
                    <label for="num-words-input">Number of words to generate:</label>
                    <input type="number" id="num-words" name="num_words" min="1" max="10" value="2">
                    <br>
                    <button type="submit" id="predict-btn">Predict</button>
                </form>
                <div id="output"></div>
                <script>
                    const predictForm = document.getElementById("predict-form");
                    predictForm.addEventListener("submit", async (event) => {
                        event.preventDefault(); // Ngăn chặn chuyển hướng trang mặc định của form
                        const text = document.getElementById("text-area").value;
                        const numWords = document.getElementById("num-words").value;
                        const response = await fetch("/predict", {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json"
                            },
                            body: JSON.stringify({ text, num_words: numWords })
                        });
                        const data = await response.text();
                        document.getElementById("output").innerHTML = data;
                    });
                </script>
            </body>
            </center>
            </html>
        `;
        res.send(html);
    } catch (error) {
        console.error(error);
        res.status(500).send("Failed to load the model.");
    }
});


app.listen(3001, () => {
    predict_function();

    console.log("Server is running...");
});
