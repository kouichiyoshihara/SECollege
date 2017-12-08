<?php

function auth($name, $pass) {

	$dsn = "mysql:host=localhost;dbname=test";
	$dbuser = "test";
	$dbpass = "test";

	try {

		$db = new PDO($dsn, $dbuser, $dbpass);
		$stmt = $db->prepare("SELECT * FROM users WHERE name = :name AND pass = :pass");
		$stmt->execute([':name' => $name, ':pass' => $pass]);
		
		if ($stmt->fetch()) {
			return true;
		}
		return false;

		$db = null;

	} catch (PDOException $e) {
	    echo $e->getMessage();
	}
}


