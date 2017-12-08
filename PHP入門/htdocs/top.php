<?php
	session_start();
	if (empty($_SESSION['auth'])) { // 未ログイン
		// ログイン・ページに転送
		header("Location: login.php");
		exit();
	}
?>

<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8">
		<title>Top</title>
	</head>
	<body>
		<h1>Welcome!</h1>
		<a href="logout.php">Logout</a>
	</body>
</html>