
from __future__ import print_function
from __future__ import division
import os
import marshal

def recursive_zip(entity, zipfileobject):
    # print(entity)
    # is it a file?
    if os.path.isfile(entity):
        zipfileobject.write(entity)
        return

    # then it must be a folder!
    for element in os.listdir(entity):
        # print(entity)
        if os.path.isfile(os.path.join(entity, element)):  # is it a regular file?
            # print('in file')
            zipfileobject.write(os.path.join(entity, element))
        elif os.path.isdir(os.path.join(entity, element)):  # or is it a folder?
            # then we must go recursively through each folder inside it
            recursive_zip(os.path.join(entity, element), zipfileobject)


def recursive_marshal(entity, marshal_object):
    # print(entity)
    # is it a file?
    if os.path.isfile(entity):
        # marshal_object.write(entity)
        marshal.dump(entity, marshal_object)
        return

    # then it must be a folder!
    for element in os.listdir(entity):
        # print(entity)
        if os.path.isfile(os.path.join(entity, element)):  # is it a regular file?
            # print('in file')
            marshal.dump(os.path.join(entity, element), marshal_object)
        elif os.path.isdir(os.path.join(entity, element)):  # or is it a folder?
            # then we must go recursively through each folder inside it
            recursive_marshal(os.path.join(entity, element), marshal_object)


def entity_to_dict(entity, dictionary):
    """
        let's convert our given file or directory to a dictionary, where:
            keys ==> file paths
            values ==> their actual data
        :argument
        :entity ==> the object that is to be sent over the network
        :dictionary ==> where we want it stored
    """
    if os.path.isfile(entity):
        # marshal_object.write(entity)
        dictionary[entity] = entity
        return

    # then it must be a folder!
    for element in os.listdir(entity):
        # print(entity)
        if os.path.isfile(os.path.join(entity, element)):  # is it a regular file?
            # print('in file')
            # marshal.dump(os.path.join(entity, element), marshal_object)
            pass
        elif os.path.isdir(os.path.join(entity, element)):  # or is it a folder?
            # then we must go recursively through each folder inside it
            # recursive_zip(os.path.join(entity, element), marshal_object)
            pass
    pass


