import { ChangeDetectorRef, Component, OnDestroy, OnInit } from '@angular/core';
import { Paginator } from "../../../paginator/paginator";
import { SearchBar } from "../../../search-bar/search-bar";
import { UsersTable } from '../users-table/users-table';
import { TranslatePipe } from '@ngx-translate/core';
import { UsersService } from '../../../../services/users/users-service';
import { UserResponseDto } from '../../../../core/api/models/user-response-dto';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { UserUpdatedService } from '../../../../services/user-updated/user-updated.service';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-inspect-users',
  imports: [Paginator, SearchBar, UsersTable, TranslatePipe],
  templateUrl: './inspect-users.html',
  styleUrl: './inspect-users.css',
})
export class InspectUsers implements OnInit, OnDestroy {
  searchTerm: string = '';
  readonly pageSize: number = 10;
  page: number = 0;
  hasMoreUsers: boolean = false;

  users: UserResponseDto[] = [];

  usersUpdatedSubscription!: Subscription;

  constructor(
    private readonly cdr: ChangeDetectorRef,
    private readonly usersService: UsersService,
    private readonly usersUpdatedService: UserUpdatedService,
  ) {}

  public ngOnInit(): void {
    this.loadUsers();
    this.usersUpdatedSubscription = this.usersUpdatedService.connect().subscribe({
      next: () => {
        this.loadUsers();
      }
    });
  }

  public debouncedSearch(searchTerm: string): void {
    this.searchTerm = searchTerm;
    this.page = 0;
    this.loadUsers();
  }

  public onPageChange(page: number): void {
    this.page = page;
    this.loadUsers();
  }

  public loadUsers(): void {
    this.usersService.get_all_users(this.searchTerm, this.page, this.pageSize).subscribe({
      next: (data: UserResponseDto[]): void => {
        this.users = data;
        this.hasMoreUsers = data.length >= this.pageSize;
        this.cdr.detectChanges();
      },
      error: (error: HttpErrorResponse) => {
        throw new Error(`Failed to load users: ${error.message}`);
      },
    });
  }

  public ngOnDestroy(): void {
    this.usersUpdatedService.disconnect();
    this.usersUpdatedSubscription.unsubscribe();
  }
}
