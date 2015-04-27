#VM Configuration

##Global Virtual Box Settings
Check VirtualBox Host-Only config by going to `File > Preferences > Network > Host-Only Networks`. Double click `vboxnet0` and ensure that the IPv4 Address is set to `192.168.56.1`.

##Rserve Config
Enable remote access with the following:
```bash
sudo touch /etc/Rserv.conf
echo "remote enable" >> /etc/Rserv.conf
```

Then, open port 6311 for tcp access:
```bash
sudo iptables -A INPUT -p tcp -m tcp --dport 6311 -j ACCEPT
sudo /etc/init.d/iptables save
```

##Configure R to use SparkR library
Need to update `.Rprofile` so the `library` function will look in the correct place:
```bash
cat >> /home/cloudera/.Rprofile <<EOT
Sys.setenv(SPARK_HOME="/home/cloudera/sparkR")

.libPaths(c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib"), .libPaths() )
EOT
```

##Add Hive configuration to Spark
```bash
cp /etc/hive/conf/hive-site.xml /home/cloudera/sparkR/conf/hive-site.xml
```

##(Optional) Disable default Spark service from running at startup
The following commands will disable the default cloudera settings that will spin up Spark 1.2 at startup. This isn't strictly necessary, but there could be some complications from running two versions of Spark in parallel.
```bash
sudo chkconfig spark-history-server off
sudo chkconfig spark-master off
sudo chkconfig spark-worker off
```
