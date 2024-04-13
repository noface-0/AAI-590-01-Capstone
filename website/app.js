const express = require('express');
const AWS = require('aws-sdk');
const path = require('path');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());
app.use(express.static('public'));

const s3 = new AWS.S3({
    accessKeyId: 'access-key',
    secretAccessKey: 'access-key',
    region: 'us-east-2',
    signatureVersion: 'v4',
});

const Bucket = 'capstone-data90210/uploaded_files';

app.post('/generate-presigned-url', (req, res) => {
    const { filename, filetype } = req.body;
    const params = {
        Bucket: Bucket,
        Key: filename,
        ContentType: filetype
    };

    console.log(params);

    s3.getSignedUrl('putObject', params, (err, url) => {
        if (err) {
            console.log(err);
            res.status(500).send('Error creating signed URL');
            return;
        }
        res.json({ url });
    });
});

app.listen(3000, () => {
    console.log('Server is running on http://localhost:3000');
});
