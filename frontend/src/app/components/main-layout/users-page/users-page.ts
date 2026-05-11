import { Component } from '@angular/core';
import { InspectUsers } from './inspect-users/inspect-users';

@Component({
  selector: 'app-users-page',
  imports: [InspectUsers],
  templateUrl: './users-page.html',
  styleUrl: './users-page.css',
})
export class UsersPage {}
