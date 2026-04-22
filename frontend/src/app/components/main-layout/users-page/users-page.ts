import { Component } from '@angular/core';
import { UsersTable } from './users-table/users-table';
import { Paginator } from '../../paginator/paginator';
import { SearchBar } from '../../search-bar/search-bar';
import { TranslatePipe } from '@ngx-translate/core';
import { InspectUsers } from './inspect-users/inspect-users';

@Component({
  selector: 'app-users-page',
  imports: [InspectUsers],
  templateUrl: './users-page.html',
  styleUrl: './users-page.css',
})
export class UsersPage {}
