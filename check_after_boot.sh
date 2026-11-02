#!/bin/sh
# Power cycle ke baad ek hi nazar me: node utha, origin kaun sa liya, aur
# sheet us point ko cover karta hai ya nahi.
#
# Ye teen sawaal hai jo har boot par galat ho sakte hai: node start hua ya
# nahi, SCR_USER padha gaya ya FC der se utha (to origin receiver se aa
# jata hai, jo 13 Oct ko galat site utha laya tha), aur sheet us point ko
# cover karta hai ya poori flight bina map fix ke jayegi.
echo "== node =="
systemctl is-active drone-vps
echo "== origin aur sheet =="
grep -aE "SCR_USER1/2:|GPS_PRIMARY =|origin from|covers this point|loaded sheet" \
    /data/DRONE/flight_logs/vps_stdout.log | tail -6
echo "== abhi ki haalat =="
cat /run/drone/vps.alive 2>/dev/null || echo "  liveness file nahi hai - node abhi chalu nahi hua"
