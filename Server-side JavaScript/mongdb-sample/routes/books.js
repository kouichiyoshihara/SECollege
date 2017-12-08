var express = require('express');
var router = express.Router();

var model = require('../model/book');

router.get('/', function(req, res /*, next*/) {
    model.Book.find(function(err, books) {
        res.render('books', {books: books});
    });
});

module.exports = router;
