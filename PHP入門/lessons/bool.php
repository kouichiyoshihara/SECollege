<?php

// echo(true); // => 1
// echo(false); // 何も表示されない
// var_export(true); // => true;
// var_export(false); // => false

// if (0) { // false
// if (0.0) { // false
// if (123) { // true
// if ("") { // false
// if ("abc") { // true
// if (1 == "1") { // true
if (1 === "1") { // false
	echo "true";
} else {
	echo "false";
}