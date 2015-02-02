__author__ = 'eddiexie'

class DataPiece():
    def __init__(self, text, lat, lng, location, user_id, created_time, source):
        self.text = text
        self.lat = lat
        self.lng = lng
        self.location = location
        self.user_id = user_id
        self.created_time = created_time
        self.source = source

    def __repr__(self):
        return "text: %s\nlat: %s\nlon: %s\nlocation: %s\nuser_id: %s\ncreated time: %s\nsource: %s\n" % self.get_repr_data_ascii() 

    def get_repr_data_ascii(self):
        return (self.text.encode('ascii', 'replace'), self.lat, self.lng, self.location.encode('ascii', 'replace'), 
                self.user_id, self.created_time, self.source)

    def get_data(self):
        return self.text, self.lat, self.lng, self.location, self.user_id, self.created_time, self.source

    def get_coordinate_pair(self):
        return (float(self.lat), float(self.lng))

    def get_info(self):
        return u" ".join([self.text, self.location])

    def get_data_as_tabed_unicode_line(self):
        return u'\t'.join(list(self.get_data())) + u'\n'

    def get_text(self):
        return self.text

    def get_lat(self):
        return self.lat

    def get_lng(self):
        return self.lng

    def get_location(self):
        return self.location

    def get_user_id(self):
        return self.user_id

    def get_created_time(self):
        return self.created_time

    def get_source(self):
        return self.source
        

