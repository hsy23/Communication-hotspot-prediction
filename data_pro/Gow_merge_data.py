import pandas as pd
import time, datetime

checkins_path = '../other_data/gowalla/gowalla_checkins.csv'
friend_path = '../other_data/gowalla/gowalla_friendship.csv'
spots_path = '../other_data/gowalla/gowalla_spots_subset1.csv'
ghash_code_index = 'bcfguvyz89destwx2367kmqr0145hjnp'

ghash_num_index = range(32)
geo_dict = dict(zip(ghash_code_index, ghash_num_index))

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

checkins = pd.read_csv(checkins_path)
with open(spots_path, 'rb') as f:
    spots = pd.read_csv(f, usecols=['id', 'lng', 'lat', 'spot_categories'])

merged_data = pd.merge(checkins, spots, left_on='placeid', right_on='id', sort=False)
merged_data = merged_data.iloc[:, [0, 1, 2, 4, 5, 6]]
merged_data['datetime'] = merged_data['datetime'].astype('str')


def cut_cate_name(name):
    pos = name.index(':', 8, -1)
    cate_name = name[pos+1:-2]
    return cate_name


def date2t(date):
    timeArray = time.strptime(date, "%Y-%m-%dT%H:%M:%SZ")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp


__all__ = ['encode', 'decode', 'bbox', 'neighbors']
_base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_decode_map = {}
_encode_map = {}
for i in range(len(_base32)):
    _decode_map[_base32[i]] = i
    _encode_map[i] = _base32[i]
del i


# 交线位置给左下
def encode(lat, lon, precision):
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    geohash = []
    code = []
    j = 0
    while len(geohash)<precision:
        # print(code,lat_range,lon_range,geohash)
        j += 1
        lat_mid = sum(lat_range)/2
        lon_mid = sum(lon_range)/2
        # 经度
        if lon<=lon_mid:
            code.append(0)
            lon_range[1]=lon_mid
        else:
            code.append(1)
            lon_range[0]=lon_mid
        # 纬度
        if lat<=lat_mid:
            code.append(0)
            lat_range[1] = lat_mid
        else:
            code.append(1)
            lat_range[0]=lat_mid
        #  encode
        if len(code) >= 5:
            geohash.append(_encode_map[int(''.join(map(str, code[:5])), 2)])
            code = code[5:]
    return ''.join(geohash)


def lnglat_encode(lng, lat, precision=7):
    result = encode(lat, lng, precision)
    return result


t1 = time.clock()
merged_data['spot_categories'] = merged_data['spot_categories'].apply(lambda x: cut_cate_name(x))
merged_data['t'] = merged_data['datetime'].apply(lambda x: date2t(x))
print(merged_data.head())
merged_data['geo'] = merged_data.apply(lambda x: lnglat_encode(x['lng'], x['lat']), axis=1)
t2 = time.clock()
print(merged_data.head())
print('time_used:', t2 - t1)

merged_data.to_csv("merged_data.csv")






