var mongoose = require('mongoose');

exports.Book = mongoose.model('Book', mongoose.Schema({
    isbn: String,
    title: String,
    price: Number
}));
