# 会悟 Meeting Management

This context describes the people and scheduled meetings shared by authenticated 会悟 users.

## Language

**User**:
An authenticated person with a 会悟 account.
_Avoid_: Account holder, member

**Profile**:
The User-managed display name, avatar, email, and phone shown by the current-account menu. Role, status, username, and account ID are not Profile fields and cannot be changed by an ordinary User.
_Avoid_: User settings, account permissions

**Colleague**:
A User explicitly saved in another User's colleague list and therefore eligible to be selected for that User's meetings.
_Avoid_: Friend, contact

**Reservation**:
A scheduled meeting with a title, time range, location, Organizer, and zero or more Participants.
_Avoid_: Calendar event, booking

**Organizer**:
The User who creates a Reservation and is its sole manager unless an administrator intervenes. The Organizer can always see the Reservation and is not duplicated in its Participant list.
_Avoid_: Owner, creator

**Participant**:
A Colleague selected by the Organizer for a Reservation. A Participant can see the Reservation on their calendar but cannot manage it.
_Avoid_: Invitee, attendee
