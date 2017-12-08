var mongoose = require('mongoose');

var db = mongoose.connection;

db.on('error', console.error.bind(console, 'connection error:'));
db.once('connected', function () {
   console.log('Connected to MongoDB');
});

mongoose.connect('mongodb://localhost/books', {useMongoClient: true});
