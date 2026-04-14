#pragma once

enum ItemType {
    LightingStorm,
    EMPBlaster,
    Deflectors,
    EmergencyEvasion,
    Count,
};

int get_item_cd(ItemType type);

int get_item_time(ItemType type);

struct Item {
    // Weapon cd
    int cd;
    // Weapon duration
    int duration;
    int x, y;
    Item(int _cd, int _duration, int _x, int _y): cd(_cd), duration(_duration), x(_x), y(_y) {
    }
    Item(ItemType it, int _x, int _y): cd(get_item_cd(it)), duration(get_item_time(it)), x(_x), y(_y) {
    }
};