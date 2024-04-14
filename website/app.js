const express = require('express');
const AWS = require('aws-sdk');
const multer = require('multer'); 
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

const upload = multer();

app.post('/submit-form', upload.none(), (req, res) => {
    const formData = req.body;
    const fileName = `form-data-${Date.now()}.json`;
    const contentType = 'application/json';
    const content = JSON.stringify(formData);

    const params = {
        Bucket: Bucket,
        Key: fileName,
        Body: content,
        ContentType: contentType
    };

    s3.upload(params, function(err, data) {
        if (err) {
            console.error("Error uploading data: ", err);
            res.status(500).send('Error uploading form data');
        } else {
            console.log("Successfully uploaded data to " + Bucket + "/" + fileName);
            res.send('Form data uploaded successfully!');
        }
    });
});

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
