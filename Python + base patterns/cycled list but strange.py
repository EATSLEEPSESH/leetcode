class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution(object):
    def mergeTwoLists(self, list1, list2):
        if list1.val <= list2.val:
            dummy = ListNode(0, list1)
            list1 = list1.next
        else:
            dummy = ListNode(0, list2)
            list2 = list2.next
        while not list1.next or not list2.next:
            if list1.val <= list2.val:
                dummy.next = list1.next
            else:
                dummy.next = list2.next

        return dummy.next
