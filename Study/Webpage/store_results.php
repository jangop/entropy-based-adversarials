#!/usr/local/cgi-bin/php
<?php
// Generate file name
$filename = 'data/'.time().'.json';

// Write request body to file
file_put_contents($filename, 'DATA' . file_get_contents("php://input"));
?>
